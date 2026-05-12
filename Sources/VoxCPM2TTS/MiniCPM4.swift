import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import MLXCommon

// MARK: - RMSNorm

public final class RMSNorm: Module {
    @ParameterInfo public var weight: MLXArray
    public let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([dimensions]))
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

// MARK: - Rotary Embedding

public final class MiniCPMLongRoPE: Module {
    @ParameterInfo var invFreq: MLXArray
    private let scalingFactor: Float
    @ParameterInfo var shortFactor: MLXArray
    @ParameterInfo var longFactor: MLXArray
    private let originalMaxPositionEmbeddings: Int

    public init(config: LMConfig) {
        let headDim = config.kvChannels ?? (config.hiddenSize / config.numAttentionHeads)
        let halfDim = headDim / 2

        let exponents = MLXArray(0..<Int32(halfDim)).asType(.float32)
            / MLXArray(Float(halfDim))
        self._invFreq = ParameterInfo(
            wrappedValue: exp(exponents * (-log(MLXArray(Float(config.ropeTheta))))),
            key: "inv_freq"
        )

        let ropeScaling = config.ropeScaling ?? RopeScalingConfig()
        let short = ropeScaling.shortFactor.isEmpty
            ? Array(repeating: 1.0, count: halfDim)
            : ropeScaling.shortFactor
        let long = ropeScaling.longFactor.isEmpty
            ? Array(repeating: 1.0, count: halfDim)
            : ropeScaling.longFactor
        self._shortFactor = ParameterInfo(wrappedValue: MLXArray(short).asType(.float32), key: "short_factor")
        self._longFactor = ParameterInfo(wrappedValue: MLXArray(long).asType(.float32), key: "long_factor")
        self.originalMaxPositionEmbeddings = max(1, ropeScaling.originalMaxPositionEmbeddings)

        let scale = Double(max(config.maxPositionEmbeddings, 1))
            / Double(max(config.originalMaxPositionEmbeddings, 1))
        self.scalingFactor = Float(sqrt(1.0 + log(max(scale, 1.0))
            / log(Double(max(config.originalMaxPositionEmbeddings, 2)))))
        super.init()
    }

    public func callAsFunction(_ positionIds: MLXArray) -> (MLXArray, MLXArray) {
        let seqLen = Int(positionIds.max().item(Int32.self)) + 1
        let factors = seqLen > originalMaxPositionEmbeddings ? longFactor : shortFactor

        let t = MLXArray(0..<Int32(seqLen)).asType(.float32)
        let freqs = (t.expandedDimensions(axis: 1)
            * (1.0 / factors.expandedDimensions(axis: 0)))
            * invFreq.expandedDimensions(axis: 0)
        let emb = concatenated([freqs, freqs], axis: -1)

        let cos = cos(emb) * MLXArray(scalingFactor)
        let sin = sin(emb) * MLXArray(scalingFactor)
        return (cos[positionIds], sin[positionIds])
    }
}

@inline(__always)
private func rotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.shape.last! / 2
    let parts = split(x, indices: [half], axis: x.ndim - 1)
    return concatenated([parts[1] * -1, parts[0]], axis: x.ndim - 1)
}

@inline(__always)
private func applyRotaryPosEmb(
    q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray
) -> (MLXArray, MLXArray) {
    let cos = cos.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let sin = sin.expandedDimensions(axis: 0).expandedDimensions(axis: 2)
    let qEmbed = (q * cos) + (rotateHalf(q) * sin)
    let kEmbed = (k * cos) + (rotateHalf(k) * sin)
    return (qEmbed, kEmbed)
}

// MARK: - Attention / MLP

public final class MiniCPMAttention: Module {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear
    @ModuleInfo(key: "q_norm") public var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") public var kNorm: RMSNorm

    public init(config: LMConfig) {
        self.numHeads = config.numAttentionHeads
        self.numKVHeads = config.numKeyValueHeads
        self.headDim = config.kvChannels ?? (config.hiddenSize / config.numAttentionHeads)
        self.scale = 1.0 / sqrt(Float(headDim))

        let qDim = numHeads * headDim
        let kvDim = numKVHeads * headDim

        self._qProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, qDim, bias: false))
        self._kProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, kvDim, bias: false))
        self._vProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, kvDim, bias: false))
        self._oProj = ModuleInfo(wrappedValue: Linear(qDim, config.hiddenSize, bias: false))
        self._qNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps))
        self._kNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: headDim, eps: config.rmsNormEps))

        super.init()
    }

    public func callAsFunction(
        _ hiddenStates: MLXArray,
        rope: MiniCPMLongRoPE?,
        cache: (MLXArray, MLXArray)? = nil,
        isCausal: Bool = true
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let batch = hiddenStates.dim(0)
        let seqLen = hiddenStates.dim(1)

        var queries = qProj(hiddenStates)
        var keys = kProj(hiddenStates)
        var values = vProj(hiddenStates)

        queries = queries.reshaped(batch, seqLen, numHeads, headDim)
        keys = keys.reshaped(batch, seqLen, numKVHeads, headDim)
        values = values.reshaped(batch, seqLen, numKVHeads, headDim)

        queries = qNorm(queries)
        keys = kNorm(keys)

        if let rope {
            let offset = cache?.0.dim(2) ?? 0
            let positionIds = MLXArray(0..<Int32(seqLen)) + Int32(offset)
            let (cos, sin) = rope(positionIds)
            (queries, keys) = applyRotaryPosEmb(q: queries, k: keys, cos: cos, sin: sin)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        var cachedKeys = keys
        var cachedValues = values
        if let (prevKeys, prevValues) = cache {
            cachedKeys = concatenated([prevKeys, keys], axis: 2)
            cachedValues = concatenated([prevValues, values], axis: 2)
        }

        let mask: MLXFast.ScaledDotProductAttentionMaskMode
        if !isCausal {
            mask = .none
        } else if seqLen <= 1 && cache == nil {
            mask = .none
        } else {
            let kvLen = cachedKeys.dim(2)
            let pastLen = kvLen - seqLen
            let causal = MLXArray.tri(seqLen, m: kvLen, k: pastLen, type: Float.self) - 1
            let additiveMask = causal * -Float.greatestFiniteMagnitude
            mask = .array(additiveMask.reshaped(1, 1, seqLen, kvLen).asType(hiddenStates.dtype))
        }

        let attn = SDPA.attendAndMerge(
            qHeads: queries, kHeads: cachedKeys, vHeads: cachedValues,
            scale: scale, mask: mask)
        let output = oProj(attn)
        return (output, (cachedKeys, cachedValues))
    }
}

public final class MiniCPMMLP: Module {
    @ModuleInfo(key: "gate_proj") public var gateProj: Linear
    @ModuleInfo(key: "up_proj") public var upProj: Linear
    @ModuleInfo(key: "down_proj") public var downProj: Linear

    public init(config: LMConfig) {
        self._gateProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.intermediateSize, bias: false))
        self._upProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, config.intermediateSize, bias: false))
        self._downProj = ModuleInfo(wrappedValue: Linear(config.intermediateSize, config.hiddenSize, bias: false))
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

public final class MiniCPMDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") public var selfAttn: MiniCPMAttention
    @ModuleInfo public var mlp: MiniCPMMLP
    @ModuleInfo(key: "input_layernorm") public var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayerNorm: RMSNorm
    public let scaleDepth: Float
    public let useMup: Bool
    public let numHiddenLayers: Int

    public init(config: LMConfig) {
        self._selfAttn = ModuleInfo(wrappedValue: MiniCPMAttention(config: config), key: "self_attn")
        self._mlp = ModuleInfo(wrappedValue: MiniCPMMLP(config: config))
        self._inputLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "input_layernorm")
        self._postAttentionLayerNorm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps), key: "post_attention_layernorm")
        self.scaleDepth = config.scaleDepth
        self.useMup = config.useMup
        self.numHiddenLayers = config.numHiddenLayers
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        rope: MiniCPMLongRoPE?,
        cache: (MLXArray, MLXArray)? = nil,
        isCausal: Bool = true
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let residual = x
        let normed = inputLayerNorm(x)
        let (attnOut, newCache) = selfAttn(normed, rope: rope, cache: cache, isCausal: isCausal)
        let residualScale = useMup ? (scaleDepth / sqrt(Float(numHiddenLayers))) : 1.0

        var h = residual + (attnOut * MLXArray(residualScale))
        let mlpOut = mlp(postAttentionLayerNorm(h)) * MLXArray(residualScale)
        h = h + mlpOut
        return (h, newCache)
    }
}

public final class MiniCPMModel: Module {
    public let config: LMConfig

    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding?
    @ModuleInfo public var layers: [MiniCPMDecoderLayer]
    @ModuleInfo public var norm: RMSNorm
    @ModuleInfo public var rope: MiniCPMLongRoPE?

    public init(_ config: LMConfig) {
        self.config = config
        if config.vocabSize > 0 {
            self._embedTokens = ModuleInfo(
                wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
                key: "embed_tokens"
            )
        } else {
            self._embedTokens = ModuleInfo(wrappedValue: nil, key: "embed_tokens")
        }
        self._layers = ModuleInfo(wrappedValue: (0..<config.numHiddenLayers).map { _ in MiniCPMDecoderLayer(config: config) })
        self._norm = ModuleInfo(wrappedValue: RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps))
        self._rope = ModuleInfo(wrappedValue: config.noRope ? nil : MiniCPMLongRoPE(config: config), key: "rope")
        super.init()
    }

    public func callAsFunction(
        inputsEmbeds: MLXArray? = nil,
        inputIds: MLXArray? = nil,
        mask: MLXArray? = nil,
        cache: [(MLXArray, MLXArray)]? = nil,
        isCausal: Bool = true
    ) -> (MLXArray, [(MLXArray, MLXArray)]) {
        let hiddenStates: MLXArray
        if let inputsEmbeds {
            hiddenStates = inputsEmbeds
        } else if let inputIds {
            guard let embedTokens else {
                fatalError("MiniCPMModel called with inputIds but no embed_tokens layer")
            }
            hiddenStates = embedTokens(inputIds)
        } else {
            fatalError("MiniCPMModel requires inputsEmbeds or inputIds")
        }

        let rope = self.rope

        var h = hiddenStates
        var newCaches: [(MLXArray, MLXArray)] = []
        newCaches.reserveCapacity(layers.count)

        for (idx, layer) in layers.enumerated() {
            let layerCache = cache?[idx]
            let (nextH, nextCache) = layer(h, rope: rope, cache: layerCache, isCausal: isCausal)
            h = nextH
            newCaches.append(nextCache)
        }

        h = norm(h)
        return (h, newCaches)
    }
}

// MARK: - VoxCPM Local Encoder

public final class VoxCPMLocEnc: Module {
    public let config: LMConfig

    public var specialToken: MLXArray
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo public var encoder: MiniCPMModel

    public init(config: LMConfig, inputDim: Int = 64) {
        self.config = config
        self.specialToken = MLXRandom.normal([1, 1, 1, config.hiddenSize]).asType(.float32)
        self._inProj = ModuleInfo(wrappedValue: Linear(inputDim, config.hiddenSize, bias: true), key: "in_proj")
        self._encoder = ModuleInfo(wrappedValue: MiniCPMModel(config))
        super.init()
    }

    public func loadSpecialToken(_ token: MLXArray) {
        self.specialToken = token
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let steps = x.dim(1)
        let patches = x.dim(2)

        var h = inProj(x)
        let special = repeated(repeated(specialToken, count: batch, axis: 0), count: steps, axis: 1)
        h = concatenated([special, h], axis: 2)
        h = h.reshaped(batch * steps, patches + 1, -1)

        let (outputs, _) = encoder(inputsEmbeds: h, isCausal: false)
        let cls = outputs[0..., 0..<1, 0...].squeezed(axis: 1)
        return cls.reshaped(batch, steps, -1)
    }
}

// MARK: - VoxCPM DiT

public final class SinusoidalPosEmb: Module {
    public let dim: Int

    public init(dim: Int) {
        precondition(dim % 2 == 0)
        self.dim = dim
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, scale: Float = 1000) -> MLXArray {
        let values = x.shape.isEmpty ? x.reshaped(1) : x.asType(.float32)
        let half = dim / 2
        let embScale = log(MLXArray(10000.0)) / MLXArray(Float(half - 1))
        let freq = exp(MLXArray(0..<Int32(half)).asType(.float32) * (-embScale))
        let emb = MLXArray(scale) * values.reshaped(-1, 1) * freq.reshaped(1, -1)
        return concatenated([sin(emb), cos(emb)], axis: -1)
    }
}

public final class TimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") public var linear_1: Linear
    @ModuleInfo(key: "linear_2") public var linear_2: Linear

    public init(inChannels: Int, timeEmbedDim: Int, outDim: Int? = nil) {
        self._linear_1 = ModuleInfo(wrappedValue: Linear(inChannels, timeEmbedDim), key: "linear_1")
        self._linear_2 = ModuleInfo(wrappedValue: Linear(timeEmbedDim, outDim ?? timeEmbedDim), key: "linear_2")
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear_2(silu(linear_1(x)))
    }
}

public final class VoxCPMLocDiTV2: Module {
    public let config: LMConfig
    public let inChannels: Int

    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "cond_proj") var condProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo var decoder: MiniCPMModel

    public let timeEmbeddings: SinusoidalPosEmb
    public let timeMlp: TimestepEmbedding
    public let deltaTimeMlp: TimestepEmbedding

    public init(config: LMConfig, inChannels: Int = 64) {
        self.config = config
        self.inChannels = inChannels

        self._inProj = ModuleInfo(wrappedValue: Linear(inChannels, config.hiddenSize))
        self._condProj = ModuleInfo(wrappedValue: Linear(inChannels, config.hiddenSize))
        self._outProj = ModuleInfo(wrappedValue: Linear(config.hiddenSize, inChannels))
        self._decoder = ModuleInfo(wrappedValue: MiniCPMModel(config))
        self.timeEmbeddings = SinusoidalPosEmb(dim: config.hiddenSize)
        self.timeMlp = TimestepEmbedding(inChannels: config.hiddenSize, timeEmbedDim: config.hiddenSize)
        self.deltaTimeMlp = TimestepEmbedding(inChannels: config.hiddenSize, timeEmbedDim: config.hiddenSize)

        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray,
        mu: MLXArray,
        t: MLXArray,
        cond: MLXArray,
        dt: MLXArray
    ) -> MLXArray {
        let batch = x.dim(0)

        let xProj = inProj(x.transposed(0, 2, 1))
        let condProj = condProj(cond.transposed(0, 2, 1))
        let prefix = condProj.dim(1)

        let tEmb = timeMlp(timeEmbeddings(t))
        let dtEmb = deltaTimeMlp(timeEmbeddings(dt))
        let timeToken = (tEmb + dtEmb).expandedDimensions(axis: 1)

        let hiddenDim = xProj.dim(2)
        let muTokens = mu.reshaped(batch, -1, hiddenDim)

        let hidden = concatenated([muTokens, timeToken, condProj, xProj], axis: 1)
        let (decoded, _) = decoder(inputsEmbeds: hidden, isCausal: false)
        let trimmed = decoded[0..., (muTokens.dim(1) + 1 + prefix)..., 0...]
        let projected = outProj(trimmed)
        return projected.transposed(0, 2, 1)
    }
}

public final class UnifiedCFM: Module {
    public let inChannels: Int
    public let cfmParams: CFMConfig
    public let meanMode: Bool

    @ModuleInfo public var estimator: VoxCPMLocDiTV2

    public init(
        inChannels: Int,
        cfmParams: CFMConfig,
        estimator: VoxCPMLocDiTV2,
        meanMode: Bool = false
    ) {
        self.inChannels = inChannels
        self.cfmParams = cfmParams
        self.meanMode = meanMode
        self._estimator = ModuleInfo(wrappedValue: estimator)
        super.init()
    }

    public func solveEuler(
        _ x: MLXArray,
        tSpan: [Float],
        mu: MLXArray,
        cond: MLXArray,
        cfgValue: Float = 1.0,
        useCfgZeroStar: Bool = true
    ) -> MLXArray {
        guard tSpan.count >= 2 else { return x }

        var currentX = x
        var t = tSpan[0]
        var dt = tSpan[0] - tSpan[1]
        let zeroInitSteps = max(1, Int(Double(tSpan.count) * 0.04))

        for step in 1..<tSpan.count {
            let dphiDt: MLXArray
            if useCfgZeroStar && step <= zeroInitSteps {
                dphiDt = MLXArray.zeros(currentX.shape, dtype: currentX.dtype)
            } else {
                let batch = currentX.dim(0)
                let xIn = concatenated([currentX, currentX], axis: 0)
                let muIn = concatenated([mu, MLXArray.zeros(mu.shape, dtype: mu.dtype)], axis: 0)
                let tVal = MLXArray(Array(repeating: t, count: batch * 2)).reshaped(batch * 2)
                let dtVal = meanMode
                    ? MLXArray(Array(repeating: dt, count: batch * 2)).reshaped(batch * 2)
                    : MLXArray.zeros([batch * 2], dtype: currentX.dtype)
                let condIn = concatenated([cond, cond], axis: 0)

                let out = estimator(xIn, mu: muIn, t: tVal, cond: condIn, dt: dtVal)
                let positive = out[0..<batch, 0..., 0...]
                let negative = out[batch..<(batch * 2), 0..., 0...]

                if useCfgZeroStar {
                    let positiveFlat = positive.reshaped(batch, -1)
                    let negativeFlat = negative.reshaped(batch, -1)
                    let dot = (positiveFlat * negativeFlat).sum(axis: 1).reshaped(batch, 1, 1)
                    let sqNorm = ((negativeFlat * negativeFlat).sum(axis: 1) + MLXArray(1e-8)).reshaped(batch, 1, 1)
                    let stStar = dot / sqNorm
                    dphiDt = negative * stStar + MLXArray(cfgValue) * (positive - negative * stStar)
                } else {
                    dphiDt = negative + MLXArray(cfgValue) * (positive - negative)
                }
            }

            currentX = currentX - MLXArray(dt) * dphiDt
            t = tSpan[step]
            if step < tSpan.count - 1 {
                dt = tSpan[step] - tSpan[step + 1]
            }
        }

        return currentX
    }

    public func sample(
        mu: MLXArray,
        nTimesteps: Int,
        patchSize: Int,
        cond: MLXArray,
        temperature: Float = 1.0,
        cfgValue: Float? = nil
    ) -> MLXArray {
        let batch = mu.dim(0)
        let z = MLXRandom.normal([batch, inChannels, patchSize], dtype: mu.dtype) * MLXArray(temperature)

        let base = (0...nTimesteps).map { Float(nTimesteps - $0) / Float(nTimesteps) }
        let tSpan = base.map { t in
            t + 1.0 * (Float(cos(Double.pi / 2.0 * Double(t))) - 1.0 + t)
        }

        return solveEuler(z, tSpan: tSpan, mu: mu, cond: cond, cfgValue: cfgValue ?? cfmParams.inferenceCfgRate)
    }
}
