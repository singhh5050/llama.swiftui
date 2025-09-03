import SwiftUI
import UIKit

struct BenchmarkView: View {
    @ObservedObject var llamaState: LlamaState
    @StateObject private var promptBank = PromptBankManager()
    @State private var maxPrompts: String = "10"
    @State private var startIndex: String = "1"
    @State private var endIndex: String = "10"
    @State private var useRange: Bool = false
    @State private var maxTokensToGenerate: String = "100"
    @State private var showingResults = false
    @State private var showingShareSheet = false
    @State private var shareURL: URL?
    
    var body: some View {
        NavigationView {
            VStack(spacing: 20) {
                // Header
                VStack {
                    Text("Prompt Bank Benchmark")
                        .font(.largeTitle)
                        .fontWeight(.bold)
                    
                    Text("Test your model against \(promptBank.prompts.count) diverse prompts")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding()
                
                // Configuration
                VStack(alignment: .leading, spacing: 15) {
                    Text("Configuration")
                        .font(.headline)
                    
                    Toggle("Use Range Selection", isOn: $useRange)
                        .toggleStyle(SwitchToggleStyle())
                    
                    if useRange {
                        VStack(spacing: 10) {
                            HStack {
                                Text("Start Index:")
                                TextField("Start", text: $startIndex)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                    .keyboardType(.numberPad)
                                    .frame(width: 80)
                                
                                Text("End Index:")
                                TextField("End", text: $endIndex)
                                    .textFieldStyle(RoundedBorderTextFieldStyle())
                                    .keyboardType(.numberPad)
                                    .frame(width: 80)
                            }
                            
                            Text("Range: \(getRangeText())")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                    } else {
                        HStack {
                            Text("Max Prompts:")
                            TextField("Number of prompts", text: $maxPrompts)
                                .textFieldStyle(RoundedBorderTextFieldStyle())
                                .keyboardType(.numberPad)
                                .frame(width: 100)
                        }
                    }
                    
                    HStack {
                        Text("Max Tokens to Generate:")
                        TextField("Tokens", text: $maxTokensToGenerate)
                            .textFieldStyle(RoundedBorderTextFieldStyle())
                            .keyboardType(.numberPad)
                            .frame(width: 80)
                        
                        Spacer()
                        
                        Text("Total Available: \(promptBank.prompts.count)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                .padding()
                .background(Color.gray.opacity(0.1))
                .cornerRadius(10)
                
                // Progress
                if promptBank.isBenchmarking {
                    VStack(spacing: 12) {
                        HStack {
                            Text("Running Benchmark...")
                                .font(.headline)
                            Spacer()
                            Text("\(Int(promptBank.benchmarkProgress * 100))%")
                                .font(.headline)
                                .fontWeight(.bold)
                        }
                        
                        ProgressView(value: promptBank.benchmarkProgress)
                            .progressViewStyle(LinearProgressViewStyle(tint: .blue))
                            .scaleEffect(x: 1, y: 2, anchor: .center)
                        
                        VStack(spacing: 8) {
                            Text(getProgressText())
                                .font(.subheadline)
                                .fontWeight(.medium)
                            
                            if !promptBank.currentPromptText.isEmpty {
                                Text("Current: \(promptBank.currentPromptText)")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                                    .lineLimit(2)
                                    .multilineTextAlignment(.center)
                            }
                            
                            HStack {
                                if promptBank.averageTimePerPrompt > 0 {
                                    Label("\(String(format: "%.1f", promptBank.averageTimePerPrompt))s avg", systemImage: "clock")
                                        .font(.caption)
                                }
                                
                                if promptBank.estimatedTimeRemaining > 0 {
                                    Spacer()
                                    Label("\(formatTime(promptBank.estimatedTimeRemaining)) remaining", systemImage: "timer")
                                        .font(.caption)
                                }
                            }
                            .foregroundColor(.secondary)
                        }
                    }
                    .padding()
                    .background(Color.blue.opacity(0.1))
                    .cornerRadius(10)
                }
                
                // Results Summary
                if !promptBank.benchmarkResults.isEmpty {
                    VStack(alignment: .leading, spacing: 10) {
                        Text("Results Summary")
                            .font(.headline)
                        
                        HStack {
                            VStack(alignment: .leading) {
                                Text("Prompts Tested:")
                                Text("Avg Prefill Speed:")
                                Text("Avg Decode Speed:")
                                Text("Avg Time to First Token:")
                            }
                            .font(.caption)
                            
                            Spacer()
                            
                            VStack(alignment: .trailing) {
                                Text("\(promptBank.benchmarkResults.count)")
                                Text(String(format: "%.1f tok/s", avgPrefillSpeed))
                                Text(String(format: "%.1f tok/s", avgDecodeSpeed))
                                Text(String(format: "%.1f ms", avgTimeToFirstToken))
                            }
                            .font(.caption)
                            .fontWeight(.medium)
                        }
                        
                        if let lastResult = promptBank.benchmarkResults.last, !lastResult.generatedText.isEmpty {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Latest Generation:")
                                    .font(.caption)
                                    .fontWeight(.medium)
                                    .foregroundColor(.blue)
                                
                                Text(String(lastResult.generatedText.prefix(150)) + (lastResult.generatedText.count > 150 ? "..." : ""))
                                    .font(.caption)
                                    .lineLimit(3)
                                    .padding(6)
                                    .background(Color.blue.opacity(0.1))
                                    .cornerRadius(4)
                            }
                        }
                        
                        Button("View Detailed Results") {
                            showingResults = true
                        }
                        .buttonStyle(.borderedProminent)
                    }
                    .padding()
                    .background(Color.green.opacity(0.1))
                    .cornerRadius(10)
                }
                
                Spacer()
                
                // Action Buttons
                VStack(spacing: 15) {
                    Button(action: startBenchmark) {
                        HStack {
                            Image(systemName: "speedometer")
                            Text(promptBank.isBenchmarking ? "Benchmarking..." : "Start Benchmark")
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(promptBank.isBenchmarking ? Color.gray : Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                    .disabled(promptBank.isBenchmarking || llamaState.getLlamaContext() == nil)
                    
                    // Memory management
                    if !promptBank.benchmarkResults.isEmpty {
                        HStack(spacing: 10) {
                            Button("Clear Results") {
                                promptBank.resetBenchmark()
                            }
                            .buttonStyle(.bordered)
                            
                            Button("Clean Memory") {
                                promptBank.clearOldResults()
                            }
                            .buttonStyle(.bordered)
                        }
                    }
                    
                    Button(action: exportCSV) {
                        HStack {
                            Image(systemName: "square.and.arrow.up")
                            if promptBank.benchmarkResults.isEmpty {
                                Text("Export Empty CSV")
                            } else if promptBank.isBenchmarking {
                                Text("Export Partial Results (\(promptBank.benchmarkResults.count))")
                            } else {
                                Text("Export Complete Results (\(promptBank.benchmarkResults.count))")
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(promptBank.benchmarkResults.isEmpty ? Color.gray : Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(10)
                    }
                }
                .padding()
            }
            .padding()
            .navigationTitle("Benchmark")
            .navigationBarTitleDisplayMode(.inline)
            .onTapGesture {
                // Dismiss keyboard when tapping anywhere on the view
                UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
            }
        }
        .sheet(isPresented: $showingResults) {
            BenchmarkResultsView(results: promptBank.benchmarkResults)
        }
        .sheet(isPresented: $showingShareSheet) {
            if let url = shareURL {
                ShareSheet(activityItems: [url])
            }
        }
        .alert("No Model Loaded", isPresented: .constant(llamaState.getLlamaContext() == nil && !promptBank.isBenchmarking)) {
            Button("OK") { }
        } message: {
            Text("Please load a model before running benchmarks.")
        }
    }
    
    private var avgPrefillSpeed: Double {
        let speeds = promptBank.benchmarkResults.map { $0.prefillSpeed }.filter { $0 > 0 }
        return speeds.isEmpty ? 0 : speeds.reduce(0, +) / Double(speeds.count)
    }
    
    private var avgDecodeSpeed: Double {
        let speeds = promptBank.benchmarkResults.map { $0.decodeSpeed }.filter { $0 > 0 }
        return speeds.isEmpty ? 0 : speeds.reduce(0, +) / Double(speeds.count)
    }
    
    private var avgTimeToFirstToken: Double {
        let times = promptBank.benchmarkResults.map { $0.timeToFirstToken }.filter { $0 > 0 }
        return times.isEmpty ? 0 : times.reduce(0, +) / Double(times.count)
    }
    
    private func startBenchmark() {
        Task {
            let maxTokens = max(10, Int(maxTokensToGenerate) ?? 100)
            
            if useRange {
                let start = max(1, Int(startIndex) ?? 1) - 1 // Convert to 0-based index
                let end = min(promptBank.prompts.count, Int(endIndex) ?? 10) - 1 // Convert to 0-based index
                await promptBank.runBenchmarkRange(llamaState: llamaState, startIndex: start, endIndex: end, maxTokens: maxTokens)
            } else {
                let maxPromptsInt = Int(maxPrompts) ?? 10
                await promptBank.runBenchmark(llamaState: llamaState, maxPrompts: maxPromptsInt, maxTokens: maxTokens)
            }
        }
    }
    
    private func getRangeText() -> String {
        let start = max(1, Int(startIndex) ?? 1)
        let end = min(promptBank.prompts.count, Int(endIndex) ?? 10)
        let count = max(0, end - start + 1)
        return "Prompts \(start)-\(end) (\(count) total)"
    }
    
    private func getProgressText() -> String {
        if useRange {
            let start = max(1, Int(startIndex) ?? 1)
            let currentAbsolute = start + promptBank.currentBenchmarkIndex
            let end = min(promptBank.prompts.count, Int(endIndex) ?? 10)
            return "Prompt \(currentAbsolute) (Range: \(start)-\(end))"
        } else {
            let total = Int(maxPrompts) ?? 10
            return "Prompt \(promptBank.currentBenchmarkIndex + 1) of \(total)"
        }
    }
    
    private func exportCSV() {
        if let url = promptBank.saveCSVToDocuments() {
            shareURL = url
            showingShareSheet = true
        }
    }
    
    private func formatTime(_ seconds: TimeInterval) -> String {
        if seconds < 60 {
            return String(format: "%.0fs", seconds)
        } else if seconds < 3600 {
            return String(format: "%.0fm %.0fs", seconds / 60, seconds.truncatingRemainder(dividingBy: 60))
        } else {
            return String(format: "%.0fh %.0fm", seconds / 3600, (seconds.truncatingRemainder(dividingBy: 3600)) / 60)
        }
    }
}

struct BenchmarkResultsView: View {
    let results: [BenchmarkResult]
    @Environment(\.dismiss) private var dismiss
    @State private var expandedItems: Set<UUID> = []
    
    var body: some View {
        NavigationView {
            List(results) { result in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Prompt \(result.promptIndex + 1)")
                            .font(.headline)
                        Spacer()
                        Button(action: {
                            if expandedItems.contains(result.id) {
                                expandedItems.remove(result.id)
                            } else {
                                expandedItems.insert(result.id)
                            }
                        }) {
                            Image(systemName: expandedItems.contains(result.id) ? "chevron.down" : "chevron.right")
                                .foregroundColor(.secondary)
                        }
                    }
                    
                    Text(result.prompt)
                        .font(.caption)
                        .lineLimit(expandedItems.contains(result.id) ? nil : 2)
                        .foregroundColor(.secondary)
                    
                    if expandedItems.contains(result.id) && !result.generatedText.isEmpty {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Generated:")
                                .font(.caption)
                                .fontWeight(.medium)
                                .foregroundColor(.blue)
                            
                            Text(result.generatedText)
                                .font(.caption)
                                .padding(8)
                                .background(Color.blue.opacity(0.1))
                                .cornerRadius(6)
                        }
                    }
                    
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Label("TTFT: \(String(format: "%.1f", result.timeToFirstToken)) ms", systemImage: "clock")
                            Label("Prefill: \(String(format: "%.1f", result.prefillSpeed)) tok/s", systemImage: "arrow.up")
                        }
                        .font(.caption)
                        
                        Spacer()
                        
                        VStack(alignment: .trailing, spacing: 4) {
                            Label("Decode: \(String(format: "%.1f", result.decodeSpeed)) tok/s", systemImage: "arrow.down")
                            Label("Tokens: \(result.totalTokens)", systemImage: "textformat")
                        }
                        .font(.caption)
                    }
                }
                .padding(.vertical, 4)
            }
            .navigationTitle("Benchmark Results")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}

struct ShareSheet: UIViewControllerRepresentable {
    let activityItems: [Any]
    
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: activityItems, applicationActivities: nil)
    }
    
    func updateUIViewController(_ uiViewController: UIActivityViewController, context: Context) {
        // No updates needed
    }
}

#Preview {
    BenchmarkView(llamaState: LlamaState())
}
