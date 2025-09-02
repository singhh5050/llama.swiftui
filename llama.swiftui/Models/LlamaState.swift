import Foundation

struct Model: Identifiable {
    var id = UUID()
    var name: String
    var url: String
    var filename: String
    var status: String?
}

@MainActor
class LlamaState: ObservableObject {
    @Published var messageLog = ""
    @Published var cacheCleared = false
    @Published var downloadedModels: [Model] = []
    @Published var undownloadedModels: [Model] = []
    let NS_PER_S = 1_000_000_000.0

    private var llamaContext: LlamaContext?
    private var currentModelUrl: URL?
    private var customSystemPrompt: String?
    private var defaultModelUrl: URL? {
        Bundle.main.url(forResource: "google_gemma-3n-E4B-it-Q4_K_M", withExtension: "gguf", subdirectory: "models")
        // Bundle.main.url(forResource: "llama-2-7b-chat", withExtension: "Q2_K.gguf", subdirectory: "models")
    }

    init() {
        loadModelsFromDisk()
        Task {
            await loadDefaultModels()
        }
    }

    private func loadModelsFromDisk() {
        do {
            let documentsURL = getDocumentsDirectory()
            let modelURLs = try FileManager.default.contentsOfDirectory(at: documentsURL, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles, .skipsSubdirectoryDescendants])
            for modelURL in modelURLs {
                let modelName = modelURL.deletingPathExtension().lastPathComponent
                downloadedModels.append(Model(name: modelName, url: "", filename: modelURL.lastPathComponent, status: "downloaded"))
            }
        } catch {
            print("Error loading models from disk: \(error)")
        }
    }

    private func loadDefaultModels() async {
        do {
            try await loadModel(modelUrl: defaultModelUrl)
        } catch {
            messageLog += "Error!\n"
        }

        for model in defaultModels {
            let fileURL = getDocumentsDirectory().appendingPathComponent(model.filename)
            if FileManager.default.fileExists(atPath: fileURL.path) {

            } else {
                var undownloadedModel = model
                undownloadedModel.status = "download"
                undownloadedModels.append(undownloadedModel)
            }
        }
    }

    func getDocumentsDirectory() -> URL {
        let paths = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)
        return paths[0]
    }
    private let defaultModels: [Model] = [
        Model(name: "TinyLlama-1.1B (Q4_0, 0.6 GiB)",url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-1T-OpenOrca-GGUF/resolve/main/tinyllama-1.1b-1t-openorca.Q4_0.gguf?download=true",filename: "tinyllama-1.1b-1t-openorca.Q4_0.gguf", status: "download"),
        Model(
            name: "TinyLlama-1.1B Chat (Q8_0, 1.1 GiB)",
            url: "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf?download=true",
            filename: "tinyllama-1.1b-chat-v1.0.Q8_0.gguf", status: "download"
        ),

        Model(
            name: "TinyLlama-1.1B (F16, 2.2 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/tinyllama-1.1b/ggml-model-f16.gguf?download=true",
            filename: "tinyllama-1.1b-f16.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q4_0, 1.6 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q4_0.gguf?download=true",
            filename: "phi-2-q4_0.gguf", status: "download"
        ),

        Model(
            name: "Phi-2.7B (Q8_0, 2.8 GiB)",
            url: "https://huggingface.co/ggml-org/models/resolve/main/phi-2/ggml-model-q8_0.gguf?download=true",
            filename: "phi-2-q8_0.gguf", status: "download"
        ),

        Model(
            name: "Mistral-7B-v0.1 (Q4_0, 3.8 GiB)",
            url: "https://huggingface.co/TheBloke/Mistral-7B-v0.1-GGUF/resolve/main/mistral-7b-v0.1.Q4_0.gguf?download=true",
            filename: "mistral-7b-v0.1.Q4_0.gguf", status: "download"
        ),
        Model(
            name: "OpenHermes-2.5-Mistral-7B (Q3_K_M, 3.52 GiB)",
            url: "https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q3_K_M.gguf?download=true",
            filename: "openhermes-2.5-mistral-7b.Q3_K_M.gguf", status: "download"
        )
    ]
    func loadModel(modelUrl: URL?) async throws {
        if let modelUrl {
            messageLog += "Loading model...\n"
            
            // Get comprehensive model information
            let filename = modelUrl.lastPathComponent
            let modelInfo = ModelStopTokens.getModelInfo(for: filename)
            
            messageLog += "✅ Detected model: \(modelInfo.type)\n"
            messageLog += "🛑 Stop tokens: \(modelInfo.stopTokens.joined(separator: ", "))\n"
            messageLog += "📝 Prompt template ready\n"
            messageLog += "💬 System prompt: Clear and focused assistant\n\n"
            
            llamaContext = await try LlamaContext.create_context(path: modelUrl.path(), stopTokens: modelInfo.stopTokens)
            currentModelUrl = modelUrl
            messageLog += "🚀 Loaded model \(modelUrl.lastPathComponent)\n"
            messageLog += "Ready for inference!\n\n"

            // Assuming that the model is successfully loaded, update the downloaded models
            updateDownloadedModels(modelName: modelUrl.lastPathComponent, status: "downloaded")
        } else {
            messageLog += "Load a model from the list below\n"
        }
    }


    private func updateDownloadedModels(modelName: String, status: String) {
        undownloadedModels.removeAll { $0.name == modelName }
    }
    
    private func getCurrentModelUrl() -> URL? {
        return currentModelUrl
    }
    
    func setCustomSystemPrompt(_ prompt: String) {
        customSystemPrompt = prompt.isEmpty ? nil : prompt
        messageLog += "📝 System prompt updated\n"
    }


    func complete(text: String) async {
        guard let llamaContext else {
            return
        }

        // Check if text looks like a raw user message (no special tokens)
        let processedText: String
        if !text.contains("<|") && !text.contains("[INST]") && !text.contains("<start_of_turn>") {
            // This looks like a plain user message, format it with the system prompt
            if let modelUrl = getCurrentModelUrl() {
                let modelInfo = ModelStopTokens.getModelInfo(for: modelUrl.lastPathComponent)
                processedText = ModelStopTokens.formatPrompt(
                    template: modelInfo.promptTemplate,
                    systemPrompt: customSystemPrompt,
                    userMessage: text
                )
                messageLog += "Using formatted prompt with system instructions\n"
            } else {
                processedText = text
            }
        } else {
            // Text already has formatting, use as-is
            processedText = text
        }

        await llamaContext.completion_init(text: processedText)
        messageLog += "\(text)"

        Task.detached {
            while await !llamaContext.is_done {
                let result = await llamaContext.completion_loop()
                await MainActor.run {
                    self.messageLog += "\(result)"
                }
            }

            // Get detailed benchmark metrics
            let metrics = await llamaContext.getBenchmarkMetrics()
            let modelInfo = await llamaContext.model_info()
            
            var benchmarkMetrics = BenchmarkMetrics()
            benchmarkMetrics.timeToFirstToken = metrics.timeToFirstToken
            benchmarkMetrics.prefillLatency = metrics.prefillLatency
            benchmarkMetrics.decodeLatency = metrics.decodeLatency
            benchmarkMetrics.prefillTokens = metrics.prefillTokens
            benchmarkMetrics.decodeTokens = metrics.decodeTokens
            benchmarkMetrics.modelSize = String(format: "%.2f GiB", Double(await llamaContext.model_size()) / 1024.0 / 1024.0 / 1024.0)
            benchmarkMetrics.modelParams = String(format: "%.2f B", Double(await llamaContext.model_params()) / 1e9)
            benchmarkMetrics.calculateDerivedMetrics()

            await llamaContext.clear()

            await MainActor.run {
                self.messageLog += "\n\n" + benchmarkMetrics.formatReport()
            }
        }
    }

    func bench() async {
        guard let llamaContext else {
            return
        }

        messageLog += "\n"
        messageLog += "Running benchmark...\n"
        messageLog += "Model info: "
        messageLog += await llamaContext.model_info() + "\n"

        let t_start = DispatchTime.now().uptimeNanoseconds
        let _ = await llamaContext.bench(pp: 8, tg: 4, pl: 1) // heat up
        let t_end = DispatchTime.now().uptimeNanoseconds

        let t_heat = Double(t_end - t_start) / NS_PER_S
        messageLog += "Heat up time: \(t_heat) seconds, please wait...\n"

        // if more than 5 seconds, then we're probably running on a slow device
        if t_heat > 5.0 {
            messageLog += "Heat up time is too long, aborting benchmark\n"
            return
        }

        let result = await llamaContext.bench(pp: 512, tg: 128, pl: 1, nr: 3)

        messageLog += "\(result)"
        messageLog += "\n"
    }

    func clear() async {
        guard let llamaContext else {
            return
        }

        await llamaContext.clear()
        messageLog = ""
    }
    
    func getLlamaContext() -> LlamaContext? {
        guard let context = llamaContext else {
            print("Warning: LlamaContext is nil - model may not be loaded")
            return nil
        }
        return context
    }
}
