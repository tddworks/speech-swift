import Foundation
import MLX
import MLXNN
import AudioCommon

/// Pre-computed reference-audio conditioning for CosyVoice 3 zero-shot voice cloning.
///
/// Holds everything the flow model needs to anchor synthesis to a specific
/// speaker:
///
///   - `speakerEmbedding`: optional 192-d CAM++ global identity vector.
///   - `promptToken`: `[1, T_prompt]` Int32 FSQ codes from the S3 tokenizer
///     (25 Hz). Prepended to the LLM-emitted speech tokens so the flow's `mu`
///     stream has a continuous reference prefix.
///   - `promptFeat`: `[1, 80, T_prompt_mel]` Matcha-style log-mel of the
///     reference (50 Hz). Written into the DiT's `cond` slot for per-frame
///     timbre anchoring.
///
/// Build once per reference clip with `CosyVoiceTTSModel.extractVoiceProfile`
/// and reuse across as many `synthesize(...)` calls as you need.
public struct CosyVoiceVoiceProfile: Sendable {
    public let speakerEmbedding: [Float]?
    public let promptToken: MLXArray?
    public let promptFeat: MLXArray?

    public init(
        speakerEmbedding: [Float]? = nil,
        promptToken: MLXArray? = nil,
        promptFeat: MLXArray? = nil
    ) {
        self.speakerEmbedding = speakerEmbedding
        self.promptToken = promptToken
        self.promptFeat = promptFeat
    }
}

extension CosyVoiceTTSModel {

    /// Extract a voice profile from a reference clip.
    ///
    /// Runs three independent feature extractors in sequence:
    ///   1. Resample to 16 kHz → 128-mel Whisper log-mel → S3 tokenizer encode
    ///      → FSQ codes at 25 Hz (`promptToken`).
    ///   2. Resample to 24 kHz → 80-mel Matcha log-mel at 50 Hz (`promptFeat`).
    ///      The 50 Hz mel must satisfy `T_mel == T_token * 2` so the cond
    ///      region aligns with the upsampled mu region — caller-side
    ///      alignment is enforced by the flow's preconditions.
    ///   3. (If a CAM++ speaker model is provided) 80-mel log-mel at 16 kHz
    ///      → 192-d speaker embedding.
    ///
    /// The caller is responsible for any audio preprocessing (denoise, loudnorm,
    /// trim leading silence). The reference should be clean speech ~5-30 s long.
    ///
    /// - Parameters:
    ///   - audio: mono float samples at the source sample rate.
    ///   - sampleRate: source sample rate of `audio` (e.g. 16000 or 24000).
    ///   - speechTokenizer: loaded `SpeechTokenizerModel` (run
    ///     `CosyVoiceWeightLoader.loadSpeechTokenizer` first).
    ///   - camppSpeaker: optional CAM++ speaker model — when present, its
    ///     192-d embedding is included in the profile.
    /// - Returns: a `CosyVoiceVoiceProfile` to pass into `synthesize`.
    public func extractVoiceProfile(
        audio: [Float],
        sampleRate: Int,
        speechTokenizer: SpeechTokenizerModel,
        camppSpeaker: CamPlusPlusSpeaker? = nil
    ) throws -> CosyVoiceVoiceProfile {
        // The two mel extractors expect specific sample rates. We resample once
        // per target rate.
        let audio16k = resample(audio, from: sampleRate, to: 16_000)
        let audio24k = resample(audio, from: sampleRate, to: 24_000)

        // 1. Speech tokenizer: 128-mel @ 16 kHz → FSQ codes @ 25 Hz.
        let whisperExtractor = WhisperMelExtractor()
        let whisperMel = whisperExtractor.extract(audio16k)               // [1, 128, T_mel100]
        var promptToken = speechTokenizer.encode(mel: whisperMel)         // [1, T_token25]
        eval(promptToken)

        // Debug-only override: COSY_OVERRIDE_FSQ_CODES=<path.bin> replaces the
        // computed codes with a known-good sequence (e.g. dumped from upstream
        // s3tokenizer). Used to isolate whether the cloning regression is from
        // the speech-tokenizer math or from elsewhere in the pipeline.
        if let overridePath = ProcessInfo.processInfo.environment["COSY_OVERRIDE_FSQ_CODES"],
           let data = try? Data(contentsOf: URL(fileURLWithPath: overridePath)) {
            let count = data.count / MemoryLayout<Int32>.size
            let codes = data.withUnsafeBytes { ptr -> [Int32] in
                let buf = ptr.bindMemory(to: Int32.self)
                return Array(buf.prefix(count))
            }
            promptToken = MLXArray(codes).expandedDimensions(axis: 0)
            eval(promptToken)
            print("  [override] prompt_token from \(overridePath): \(count) codes")
        }

        // 2. Flow mel: 80-mel @ 24 kHz, 50 Hz frame rate.
        let flowExtractor = FlowMelExtractor()
        var promptFeat = flowExtractor.extract(audio24k)                  // [1, 80, T_mel50]

        // Debug override: COSY_OVERRIDE_FLOW_MEL=<path.bin> replaces the computed
        // prompt_feat. The file must be raw float32 with shape [n_mels, T] (use
        // the matching .meta.json describing shape). Used to isolate whether the
        // remaining cloning gap is from the flow mel extractor or further down.
        if let overridePath = ProcessInfo.processInfo.environment["COSY_OVERRIDE_FLOW_MEL"],
           let data = try? Data(contentsOf: URL(fileURLWithPath: overridePath)) {
            let count = data.count / MemoryLayout<Float>.size
            let floats = data.withUnsafeBytes { ptr -> [Float] in
                let buf = ptr.bindMemory(to: Float.self)
                return Array(buf.prefix(count))
            }
            // Shape is [80, T] from upstream. Reshape to [1, 80, T].
            let T = count / 80
            promptFeat = MLXArray(floats, [1, 80, T])
            eval(promptFeat)
            print("  [override] prompt_feat from \(overridePath): [1, 80, \(T)]")
        }

        // Debug dump for the Python-vs-Swift diff. Activated by
        // COSY_DEBUG_DUMP_DIR=<dir> — writes raw binary float32 / int32 files
        // and exits the calling pipeline at the natural break (caller decides).
        // Safe to leave in for now; no-op unless the env var is set.
        if let dumpDir = ProcessInfo.processInfo.environment["COSY_DEBUG_DUMP_DIR"] {
            CosyVoiceDebugDump.tryWrite(whisperMel, name: "swift_whisper_mel", in: dumpDir)
            CosyVoiceDebugDump.tryWrite(promptToken, name: "swift_fsq_codes", in: dumpDir)
            CosyVoiceDebugDump.tryWrite(promptFeat, name: "swift_flow_mel", in: dumpDir)
        }

        // Don't force-align prompt_feat length to prompt_token * tokenMelRatio.
        // Upstream's flow.inference takes whatever matcha.mel produces (1966
        // frames for a ~39 s clip) and lets it differ from prompt_token_len * 2
        // (1968 frames) by a small amount — the conds tensor is sized to the
        // FULL upsampled-mu length and has prompt_feat in `[:mel_len1]` and
        // zeros for the rest (including the 2-frame gap). The slice at the end
        // of synthesise then uses `mel_len1` (= prompt_feat.shape[2]), keeping
        // those gap frames in the generation region as transition frames.
        // Padding to 2*T_token here caused us to lose 2 mel frames off the
        // front of the generated content and confused the cond conditioning.
        eval(promptFeat)

        // 3. (Optional) CAM++ 192-d speaker embedding.
        let speakerEmbedding: [Float]? = try camppSpeaker.flatMap { spk in
            try spk.embed(audio: audio16k, sampleRate: 16_000)
        }

        return CosyVoiceVoiceProfile(
            speakerEmbedding: speakerEmbedding,
            promptToken: promptToken,
            promptFeat: promptFeat
        )
    }

    /// Convenience: synthesize with a `CosyVoiceVoiceProfile` directly.
    public func synthesize(
        text: String,
        voiceProfile: CosyVoiceVoiceProfile,
        language: String = "english",
        instruction: String = "You are a helpful assistant.",
        verbose: Bool = false
    ) -> [Float] {
        synthesize(
            text: text,
            language: language,
            instruction: instruction,
            speakerEmbedding: voiceProfile.speakerEmbedding,
            promptToken: voiceProfile.promptToken,
            promptFeat: voiceProfile.promptFeat,
            verbose: verbose
        )
    }
}

// MARK: - Debug dump (env-var gated)

/// Tiny helper that writes MLX tensors as `<name>.bin` (raw little-endian
/// float32 or int32) + `<name>.shape.json` (JSON array of dim sizes) so a
/// Python sidecar can `np.fromfile(...).reshape(shape)` and diff against
/// upstream. Activated by setting `COSY_DEBUG_DUMP_DIR=<dir>`.
enum CosyVoiceDebugDump {
    static func tryWrite(_ array: MLXArray, name: String, in dir: String) {
        let fm = FileManager.default
        try? fm.createDirectory(atPath: dir, withIntermediateDirectories: true)
        let shape = array.shape
        let shapeJSON = "[" + shape.map(String.init).joined(separator: ",") + "]"
        let dtype = array.dtype
        let dtypeStr: String
        let bytes: Data
        switch dtype {
        case .float32:
            dtypeStr = "float32"
            let flat = array.reshaped(-1).asArray(Float.self)
            bytes = flat.withUnsafeBufferPointer { Data(buffer: $0) }
        case .int32:
            dtypeStr = "int32"
            let flat = array.reshaped(-1).asArray(Int32.self)
            bytes = flat.withUnsafeBufferPointer { Data(buffer: $0) }
        default:
            // Cast bf16/fp16 to fp32 for the dump.
            dtypeStr = "float32"
            let flat = array.asType(.float32).reshaped(-1).asArray(Float.self)
            bytes = flat.withUnsafeBufferPointer { Data(buffer: $0) }
        }
        let binPath = "\(dir)/\(name).bin"
        try? bytes.write(to: URL(fileURLWithPath: binPath))
        let meta = "{\"shape\":\(shapeJSON),\"dtype\":\"\(dtypeStr)\"}\n"
        try? meta.write(toFile: "\(dir)/\(name).meta.json", atomically: true, encoding: .utf8)
        print("  [debug-dump] \(name) shape=\(shape) dtype=\(dtypeStr) → \(binPath)")
    }
}

// MARK: - Resampling

/// Linear-interpolation resampler. Good enough for the 16 kHz ↔ 24 kHz hops
/// the cloning path needs — both mel extractors are robust to the small phase
/// distortion this introduces vs a polyphase resampler.
private func resample(_ x: [Float], from src: Int, to dst: Int) -> [Float] {
    if src == dst { return x }
    let ratio = Double(dst) / Double(src)
    let outLen = Int((Double(x.count) * ratio).rounded(.down))
    guard outLen > 0 else { return [] }
    var out = [Float](repeating: 0, count: outLen)
    let step = 1.0 / ratio
    for i in 0..<outLen {
        let src_pos = Double(i) * step
        let idx = Int(src_pos)
        let frac = Float(src_pos - Double(idx))
        let a = x[idx]
        let b = idx + 1 < x.count ? x[idx + 1] : a
        out[i] = a * (1 - frac) + b * frac
    }
    return out
}
