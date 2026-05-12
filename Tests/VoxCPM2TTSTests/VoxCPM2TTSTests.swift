import XCTest
import Foundation
@testable import VoxCPM2TTS

final class VoxCPM2TTSConfigTests: XCTestCase {
    func testModelArgsRoundTripAndLoad() throws {
        var args = ModelArgs()
        args.lmConfig.hiddenSize = 1536
        args.lmConfig.numHiddenLayers = 12
        args.encoderConfig.numLayers = 3
        args.ditConfig.numLayers = 6
        args.audioVAEConfig.outSampleRate = 44_100
        args.scalarQuantizationLatentDim = 256
        args.maxLength = 4096

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let data = try encoder.encode(args)

        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let configURL = directory.appendingPathComponent("config.json")
        try data.write(to: configURL, options: .atomic)

        let loaded = try ModelArgs.load(from: directory)
        XCTAssertEqual(loaded.lmConfig.hiddenSize, 1536)
        XCTAssertEqual(loaded.lmConfig.numHiddenLayers, 12)
        XCTAssertEqual(loaded.lmConfig.kvChannels, 128)
        XCTAssertEqual(loaded.encoderConfig.numLayers, 3)
        XCTAssertEqual(loaded.encoderConfig.kvChannels, 128)
        XCTAssertEqual(loaded.ditConfig.numLayers, 6)
        XCTAssertEqual(loaded.ditConfig.kvChannels, 128)
        XCTAssertTrue(loaded.residualLMNoRope)
        XCTAssertEqual(loaded.audioVAEConfig.outSampleRate, 44_100)
        XCTAssertEqual(loaded.scalarQuantizationLatentDim, 256)
        XCTAssertEqual(loaded.maxLength, 4096)
    }

    func testModelArgsLoadsOfficialStyleConfigWithoutLegacyLmKeys() throws {
        let json = """
        {
          "lm_config": {
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "max_position_embeddings": 32768,
            "num_attention_heads": 16,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-05,
            "rope_theta": 10000,
            "kv_channels": 128,
            "rope_scaling": {
              "type": "longrope",
              "long_factor": [1.0],
              "short_factor": [1.0]
            },
            "vocab_size": 73448,
            "use_mup": false,
            "scale_emb": 12,
            "dim_model_base": 256,
            "scale_depth": 1.4
          },
          "patch_size": 4,
          "feat_dim": 64,
          "scalar_quantization_latent_dim": 512,
          "scalar_quantization_scale": 9,
          "residual_lm_num_layers": 8,
          "residual_lm_no_rope": true,
          "encoder_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
            "kv_channels": 128
          },
          "dit_config": {
            "hidden_dim": 1024,
            "ffn_dim": 4096,
            "num_heads": 16,
            "num_layers": 12,
            "kv_channels": 128,
            "mean_mode": false,
            "cfm_config": {
              "sigma_min": 1e-06,
              "solver": "euler",
              "t_scheduler": "log-norm",
              "inference_cfg_rate": 2.0
            }
          },
          "audio_vae_config": {
            "encoder_dim": 128,
            "encoder_rates": [2, 5, 8, 8],
            "latent_dim": 64,
            "decoder_dim": 2048,
            "decoder_rates": [8, 6, 5, 2, 2, 2],
            "sr_bin_boundaries": [20000, 30000, 40000],
            "sample_rate": 16000,
            "out_sample_rate": 48000
          },
          "max_length": 8192,
          "model_type": "voxcpm2"
        }
        """.data(using: .utf8)!

        let decoded = try JSONDecoder().decode(ModelArgs.self, from: json)
        XCTAssertEqual(decoded.lmConfig.hiddenSize, 2048)
        XCTAssertEqual(decoded.lmConfig.kvChannels, 128)
        XCTAssertEqual(decoded.lmConfig.noRope, false)
        XCTAssertEqual(decoded.lmConfig.originalMaxPositionEmbeddings, 32768)
        XCTAssertEqual(decoded.residualLMNoRope, true)
        XCTAssertEqual(decoded.audioVAEConfig.outSampleRate, 48_000)
    }

    func testRopeScalingCodableSnakeCase() throws {
        let scaling = RopeScalingConfig()
        var copy = scaling
        copy.shortFactor = [1.0, 2.0]
        copy.longFactor = [3.0, 4.0]
        copy.originalMaxPositionEmbeddings = 8192

        let data = try JSONEncoder().encode(copy)
        let decoded = try JSONDecoder().decode(RopeScalingConfig.self, from: data)

        XCTAssertEqual(decoded.type, "longrope")
        XCTAssertEqual(decoded.shortFactor, [1.0, 2.0])
        XCTAssertEqual(decoded.longFactor, [3.0, 4.0])
        XCTAssertEqual(decoded.originalMaxPositionEmbeddings, 8192)
    }

    func testAudioVAEConfigDefaults() {
        let config = AudioVAEConfig()
        XCTAssertEqual(config.encoderDim, 128)
        XCTAssertEqual(config.encoderRates, [2, 5, 8, 8])
        XCTAssertEqual(config.latentDim, 64)
        XCTAssertEqual(config.decoderRates, [8, 6, 5, 2, 2, 2])
        XCTAssertEqual(config.sampleRate, 16_000)
        XCTAssertEqual(config.outSampleRate, 48_000)
        XCTAssertEqual(config.srBinBoundaries, [20_000, 30_000, 40_000])
    }
}

final class VoxCPM2TTSLayerTests: XCTestCase {
    private static var hasMLXMetallib: Bool {
        let fm = FileManager.default
        let candidates = [
            Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("mlx.metallib"),
            Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("Resources/mlx.metallib"),
            Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("default.metallib"),
            Bundle.main.executableURL?.deletingLastPathComponent().appendingPathComponent("Resources/default.metallib")
        ].compactMap { $0 }

        return candidates.contains { fm.fileExists(atPath: $0.path) }
    }

    func testScalarQuantizationLayerInitializes() throws {
        try XCTSkipUnless(Self.hasMLXMetallib, "MLX metallib not available in this test environment")
        let layer = ScalarQuantizationLayer(inDim: 2, outDim: 3, latentDim: 4, scale: 9)

        XCTAssertEqual(layer.scale, 9)
        XCTAssertEqual(layer.in_proj.shape.0, 4)
        XCTAssertEqual(layer.out_proj.shape.0, 3)
    }
}
