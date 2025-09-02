import Foundation
import UIKit

struct PromptBankItem: Identifiable, Codable {
    let id = UUID()
    let prompt: String
    
    init(prompt: String) {
        self.prompt = prompt
    }
}

struct BenchmarkResult: Identifiable, Codable {
    let id = UUID()
    let promptIndex: Int
    let prompt: String
    let generatedText: String
    let timeToFirstToken: Double
    let prefillLatency: Double
    let decodeLatency: Double
    let prefillTokens: Int
    let decodeTokens: Int
    let totalTokens: Int
    let modelName: String
    let modelSize: String
    let modelParams: String
    let backend: String
    let timestamp: Date
    
    // Derived metrics
    var prefillSpeed: Double {
        prefillTokens > 0 && prefillLatency > 0 ? Double(prefillTokens) / (prefillLatency / 1000.0) : 0
    }
    
    var decodeSpeed: Double {
        decodeTokens > 0 && decodeLatency > 0 ? Double(decodeTokens) / (decodeLatency / 1000.0) : 0
    }
    
    var totalLatency: Double {
        prefillLatency + decodeLatency
    }
}

@MainActor
class PromptBankManager: ObservableObject {
    @Published var prompts: [PromptBankItem] = []
    @Published var benchmarkResults: [BenchmarkResult] = []
    @Published var isLoading = false
    @Published var currentBenchmarkIndex = 0
    @Published var benchmarkProgress: Double = 0.0
    @Published var isBenchmarking = false
    @Published var currentPromptText = ""
    @Published var estimatedTimeRemaining: TimeInterval = 0
    @Published var averageTimePerPrompt: TimeInterval = 0
    private var benchmarkStartTime: Date?
    
    // Memory management
    private let maxResultsInMemory = 1000 // Limit results to prevent memory issues
    private var resultsBatchSize = 50 // Process results in batches
    
    init() {
        loadPrompts()
    }
    
    private func loadPrompts() {
        // Try to load from app bundle first
        if let bundleURL = Bundle.main.url(forResource: "prompt_bank", withExtension: "json") {
            do {
                let data = try Data(contentsOf: bundleURL)
                let promptStrings = try JSONDecoder().decode([String].self, from: data)
                prompts = promptStrings.map { PromptBankItem(prompt: $0) }
                print("Successfully loaded \(prompts.count) prompts from app bundle")
                return
            } catch {
                print("Error loading from bundle: \(error)")
            }
        }
        
        // Try documents directory as fallback
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let documentsURL = documentsPath.appendingPathComponent("prompt_bank.json")
        
        if FileManager.default.fileExists(atPath: documentsURL.path) {
            do {
                let data = try Data(contentsOf: documentsURL)
                let promptStrings = try JSONDecoder().decode([String].self, from: data)
                prompts = promptStrings.map { PromptBankItem(prompt: $0) }
                print("Successfully loaded \(prompts.count) prompts from documents directory")
                return
            } catch {
                print("Error loading from documents: \(error)")
            }
        }
        
        // Use fallback prompts
        print("Failed to load prompt bank from any location, using fallback prompts")
        prompts = [
            PromptBankItem(prompt: "Write a short story about a robot learning to paint."),
            PromptBankItem(prompt: "Explain quantum computing in simple terms."),
            PromptBankItem(prompt: "Create a recipe for chocolate chip cookies."),
            PromptBankItem(prompt: "What are the benefits of renewable energy?"),
            PromptBankItem(prompt: "Describe a day in the life of a medieval knight."),
            PromptBankItem(prompt: "Write a Python function to calculate fibonacci numbers."),
            PromptBankItem(prompt: "What are the main differences between iOS and Android?"),
            PromptBankItem(prompt: "Explain the concept of machine learning to a beginner."),
            PromptBankItem(prompt: "Write a haiku about artificial intelligence."),
            PromptBankItem(prompt: "How do neural networks work?")
        ]
        print("Using fallback prompts: \(prompts.count)")
    }
    
    func runBenchmark(llamaState: LlamaState, maxPrompts: Int? = nil, maxTokens: Int = 100) async {
        guard !isBenchmarking else { return }
        
        isBenchmarking = true
        benchmarkResults.removeAll()
        currentBenchmarkIndex = 0
        benchmarkStartTime = Date()
        averageTimePerPrompt = 0
        estimatedTimeRemaining = 0
        
        let promptsToTest = maxPrompts.map { min($0, prompts.count) } ?? prompts.count
        
        for i in 0..<promptsToTest {
            currentBenchmarkIndex = i
            benchmarkProgress = Double(i) / Double(promptsToTest)
            
            let prompt = prompts[i]
            currentPromptText = String(prompt.prompt.prefix(100)) + (prompt.prompt.count > 100 ? "..." : "")
            
            let promptStartTime = Date()
            
            // Run the prompt and collect metrics
            if let result = await runSingleBenchmark(prompt: prompt, index: i, llamaState: llamaState, maxTokens: maxTokens) {
                // Memory management - limit results in memory
                if benchmarkResults.count >= maxResultsInMemory {
                    // Remove oldest results but keep recent ones for UI
                    let keepRecent = 100
                    benchmarkResults = Array(benchmarkResults.suffix(keepRecent))
                }
                
                benchmarkResults.append(result)
                
                // Update timing estimates
                let promptDuration = Date().timeIntervalSince(promptStartTime)
                let totalElapsed = Date().timeIntervalSince(benchmarkStartTime!)
                averageTimePerPrompt = totalElapsed / Double(i + 1)
                let remainingPrompts = promptsToTest - (i + 1)
                estimatedTimeRemaining = averageTimePerPrompt * Double(remainingPrompts)
                
                // Yield control periodically for UI responsiveness
                if i % 5 == 0 {
                    await Task.yield()
                }
            }
            
            // Memory pressure relief - longer delay every 10 prompts
            if i % 10 == 0 {
                try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 seconds
            } else {
                try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
            }
        }
        
        benchmarkProgress = 1.0
        currentPromptText = "Benchmark completed!"
        estimatedTimeRemaining = 0
        isBenchmarking = false
        
        print("Benchmark completed! Tested \(benchmarkResults.count) prompts")
    }
    
    func runBenchmarkRange(llamaState: LlamaState, startIndex: Int, endIndex: Int, maxTokens: Int = 100) async {
        guard !isBenchmarking else { return }
        guard startIndex >= 0, endIndex < prompts.count, startIndex <= endIndex else {
            print("Invalid range: \(startIndex)-\(endIndex)")
            return
        }
        
        isBenchmarking = true
        benchmarkResults.removeAll()
        currentBenchmarkIndex = 0
        benchmarkStartTime = Date()
        averageTimePerPrompt = 0
        estimatedTimeRemaining = 0
        
        let totalPrompts = endIndex - startIndex + 1
        
        for i in startIndex...endIndex {
            let relativeIndex = i - startIndex
            currentBenchmarkIndex = relativeIndex
            benchmarkProgress = Double(relativeIndex) / Double(totalPrompts)
            
            let prompt = prompts[i]
            currentPromptText = String(prompt.prompt.prefix(100)) + (prompt.prompt.count > 100 ? "..." : "")
            
            let promptStartTime = Date()
            
            // Run the prompt and collect metrics
            if let result = await runSingleBenchmark(prompt: prompt, index: i, llamaState: llamaState, maxTokens: maxTokens) {
                // Memory management - limit results in memory
                if benchmarkResults.count >= maxResultsInMemory {
                    // Remove oldest results but keep recent ones for UI
                    let keepRecent = 100
                    benchmarkResults = Array(benchmarkResults.suffix(keepRecent))
                }
                
                benchmarkResults.append(result)
                
                // Update timing estimates
                let promptDuration = Date().timeIntervalSince(promptStartTime)
                let totalElapsed = Date().timeIntervalSince(benchmarkStartTime!)
                averageTimePerPrompt = totalElapsed / Double(relativeIndex + 1)
                let remainingPrompts = totalPrompts - (relativeIndex + 1)
                estimatedTimeRemaining = averageTimePerPrompt * Double(remainingPrompts)
                
                // Yield control periodically for UI responsiveness
                if relativeIndex % 5 == 0 {
                    await Task.yield()
                }
            }
            
            // Memory pressure relief - longer delay every 10 prompts
            if relativeIndex % 10 == 0 {
                try? await Task.sleep(nanoseconds: 200_000_000) // 0.2 seconds
            } else {
                try? await Task.sleep(nanoseconds: 100_000_000) // 0.1 seconds
            }
        }
        
        benchmarkProgress = 1.0
        currentPromptText = "Benchmark completed!"
        estimatedTimeRemaining = 0
        isBenchmarking = false
        
        print("Range benchmark completed! Tested \(benchmarkResults.count) prompts (range \(startIndex+1)-\(endIndex+1))")
    }
    
    private func runSingleBenchmark(prompt: PromptBankItem, index: Int, llamaState: LlamaState, maxTokens: Int = 100) async -> BenchmarkResult? {
        // Guard against nil context
        guard let llamaContext = await llamaState.getLlamaContext() else {
            print("Error: LlamaContext is nil for prompt \(index)")
            return nil
        }
        
        do {
            // Clear previous state with error handling
            await llamaContext.clear()
            
            // Start timing
            let startTime = Date()
            
            // Guard against empty prompts
            guard !prompt.prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                print("Warning: Empty prompt at index \(index)")
                return nil
            }
            
            // Initialize completion with error handling
            await llamaContext.completion_init(text: prompt.prompt)
            
            // Run completion loop until done with timeout protection
            var generatedTokens = 0
            var generatedText = ""
            let maxIterations = maxTokens * 2 // Prevent infinite loops (2x safety margin)
            var iterations = 0
            
            while await !llamaContext.is_done && 
                  generatedTokens < maxTokens && 
                  iterations < maxIterations {
                
                let tokenResult = await llamaContext.completion_loop()
                generatedText += tokenResult
                generatedTokens += 1
                iterations += 1
                
                // Memory pressure check - yield periodically
                if iterations % 10 == 0 {
                    await Task.yield()
                }
            }
            
            // Get metrics with nil safety
            let metrics = await llamaContext.getBenchmarkMetrics()
            
            // Guard against nil or invalid metrics
            guard metrics.prefillTokens >= 0 && metrics.decodeTokens >= 0 else {
                print("Warning: Invalid metrics for prompt \(index)")
                return nil
            }
            
            let modelInfo = await llamaContext.model_info()
            let modelSize = String(format: "%.2f GiB", Double(await llamaContext.model_size()) / 1024.0 / 1024.0 / 1024.0)
            let modelParams = String(format: "%.2f B", Double(await llamaContext.model_params()) / 1e9)
            
            // Clean up generated text for CSV (remove extra whitespace, newlines)
            let cleanedGeneratedText = generatedText
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: "\n", with: " ")
                .replacingOccurrences(of: "\r", with: " ")
                .replacingOccurrences(of: "\"", with: "\"\"") // Escape quotes for CSV
            
            return BenchmarkResult(
                promptIndex: index,
                prompt: String(prompt.prompt.prefix(200)), // Show more of prompt for context
                generatedText: cleanedGeneratedText,
                timeToFirstToken: metrics.timeToFirstToken,
                prefillLatency: metrics.prefillLatency,
                decodeLatency: metrics.decodeLatency,
                prefillTokens: metrics.prefillTokens,
                decodeTokens: metrics.decodeTokens,
                totalTokens: metrics.prefillTokens + metrics.decodeTokens,
                modelName: modelInfo,
                modelSize: modelSize,
                modelParams: modelParams,
                backend: "Metal",
                timestamp: startTime
            )
            
        } catch {
            print("Error during benchmark for prompt \(index): \(error)")
            return nil
        }
    }
    
    func exportToCSV() -> String {
        let now = Date()
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .medium
        
        var csv = "# Llama.cpp Benchmark Results\n"
        csv += "# Generated: \(formatter.string(from: now))\n"
        csv += "# Total Prompts Available: \(prompts.count)\n"
        csv += "# Results Count: \(benchmarkResults.count)\n"
        csv += "# Status: \(isBenchmarking ? "In Progress" : "Complete")\n"
        if isBenchmarking {
            csv += "# Progress: \(Int(benchmarkProgress * 100))%\n"
        }
        csv += "#\n"
        csv += "Index,Prompt,Generated_Text,Time_to_First_Token_ms,Prefill_Latency_ms,Decode_Latency_ms,Prefill_Tokens,Decode_Tokens,Total_Tokens,Prefill_Speed_tps,Decode_Speed_tps,Total_Latency_ms,Model_Name,Model_Size,Model_Params,Backend,Timestamp\n"
        
        if benchmarkResults.isEmpty {
            csv += "# No results available yet\n"
        } else {
            for result in benchmarkResults {
                let escapedPrompt = result.prompt.replacingOccurrences(of: "\"", with: "\"\"")
                let escapedGeneratedText = result.generatedText.replacingOccurrences(of: "\"", with: "\"\"")
                let row = "\(result.promptIndex),\"\(escapedPrompt)\",\"\(escapedGeneratedText)\",\(result.timeToFirstToken),\(result.prefillLatency),\(result.decodeLatency),\(result.prefillTokens),\(result.decodeTokens),\(result.totalTokens),\(String(format: "%.2f", result.prefillSpeed)),\(String(format: "%.2f", result.decodeSpeed)),\(result.totalLatency),\"\(result.modelName)\",\(result.modelSize),\(result.modelParams),\(result.backend),\(ISO8601DateFormatter().string(from: result.timestamp))\n"
                csv += row
            }
        }
        
        return csv
    }
    
    func saveCSVToDocuments() -> URL? {
        let csv = exportToCSV()
        let dateFormatter = DateFormatter()
        dateFormatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let filename = "llama_benchmark_\(dateFormatter.string(from: Date())).csv"
        
        let documentsPath = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let fileURL = documentsPath.appendingPathComponent(filename)
        
        do {
            try csv.write(to: fileURL, atomically: true, encoding: .utf8)
            print("CSV saved to: \(fileURL.path)")
            return fileURL
        } catch {
            print("Error saving CSV: \(error)")
            return nil
        }
    }
    
    // Memory cleanup utilities
    func clearOldResults() {
        if benchmarkResults.count > maxResultsInMemory {
            let keepRecent = 100
            benchmarkResults = Array(benchmarkResults.suffix(keepRecent))
        }
    }
    
    func resetBenchmark() {
        isBenchmarking = false
        benchmarkProgress = 0.0
        currentBenchmarkIndex = 0
        currentPromptText = ""
        estimatedTimeRemaining = 0
        averageTimePerPrompt = 0
        benchmarkStartTime = nil
        // Optionally clear results
        benchmarkResults.removeAll()
    }
}
