import Foundation

struct BenchmarkMetrics {
    // Time to First Token (TTFT)
    var timeToFirstToken: Double = 0.0
    
    // Prefill metrics
    var prefillLatency: Double = 0.0
    var prefillTokens: Int = 0
    var prefillTokensPerSecond: Double = 0.0
    
    // Decode metrics
    var decodeLatency: Double = 0.0
    var decodeTokens: Int = 0
    var decodeTokensPerSecond: Double = 0.0
    
    // Overall metrics
    var totalTokens: Int = 0
    var totalTime: Double = 0.0
    var overallTokensPerSecond: Double = 0.0
    
    // System metrics
    var modelSize: String = ""
    var modelParams: String = ""
    var backend: String = "Metal"
    var memoryUsage: Int64 = 0
    
    // Performance scores
    var efficiency: Double = 0.0 // tokens/sec per GB of model
    var responsiveness: Double = 0.0 // 1000/TTFT (higher is better)
    
    mutating func calculateDerivedMetrics() {
        if prefillTokens > 0 && prefillLatency > 0 {
            prefillTokensPerSecond = Double(prefillTokens) / prefillLatency
        }
        
        if decodeTokens > 0 && decodeLatency > 0 {
            decodeTokensPerSecond = Double(decodeTokens) / decodeLatency
        }
        
        totalTokens = prefillTokens + decodeTokens
        totalTime = prefillLatency + decodeLatency
        
        if totalTokens > 0 && totalTime > 0 {
            overallTokensPerSecond = Double(totalTokens) / totalTime
        }
        
        if timeToFirstToken > 0 {
            responsiveness = 1000.0 / timeToFirstToken
        }
        
        // Calculate efficiency if model size is available
        if let modelSizeGB = extractModelSizeGB(from: modelSize), modelSizeGB > 0 {
            efficiency = overallTokensPerSecond / modelSizeGB
        }
    }
    
    private func extractModelSizeGB(from sizeString: String) -> Double? {
        let components = sizeString.components(separatedBy: " ")
        if components.count >= 2,
           let value = Double(components[0]),
           components[1].lowercased().contains("gib") {
            return value
        }
        return nil
    }
    
    func formatReport() -> String {
        var report = "📊 DETAILED BENCHMARK REPORT\n"
        report += String(repeating: "=", count: 50) + "\n\n"
        
        // Model Information
        report += "🤖 Model Information:\n"
        report += "   Model Size: \(modelSize)\n"
        report += "   Parameters: \(modelParams)\n"
        report += "   Backend: \(backend)\n"
        report += "   Memory Usage: \(formatBytes(memoryUsage))\n\n"
        
        // Performance Metrics
        report += "⚡ Performance Metrics:\n"
        report += "   Time to First Token (TTFT): \(String(format: "%.2f", timeToFirstToken))ms\n"
        report += "   Responsiveness Score: \(String(format: "%.1f", responsiveness))\n\n"
        
        // Prefill Phase
        report += "📝 Prefill Phase:\n"
        report += "   Latency: \(String(format: "%.2f", prefillLatency))ms\n"
        report += "   Tokens: \(prefillTokens)\n"
        report += "   Speed: \(String(format: "%.2f", prefillTokensPerSecond)) t/s\n\n"
        
        // Decode Phase
        report += "🎯 Decode Phase:\n"
        report += "   Latency: \(String(format: "%.2f", decodeLatency))ms\n"
        report += "   Tokens: \(decodeTokens)\n"
        report += "   Speed: \(String(format: "%.2f", decodeTokensPerSecond)) t/s\n\n"
        
        // Overall Performance
        report += "📈 Overall Performance:\n"
        report += "   Total Tokens: \(totalTokens)\n"
        report += "   Total Time: \(String(format: "%.2f", totalTime))ms\n"
        report += "   Average Speed: \(String(format: "%.2f", overallTokensPerSecond)) t/s\n"
        report += "   Efficiency: \(String(format: "%.2f", efficiency)) t/s/GB\n\n"
        
        // Performance Rating
        let rating = getPerformanceRating()
        report += "🏆 Performance Rating: \(rating)\n"
        
        return report
    }
    
    private func getPerformanceRating() -> String {
        if overallTokensPerSecond > 50 { return "🚀 Excellent" }
        else if overallTokensPerSecond > 25 { return "⭐ Great" }
        else if overallTokensPerSecond > 15 { return "👍 Good" }
        else if overallTokensPerSecond > 8 { return "⚠️ Fair" }
        else { return "🐌 Needs Optimization" }
    }
    
    private func formatBytes(_ bytes: Int64) -> String {
        let formatter = ByteCountFormatter()
        formatter.countStyle = .binary
        return formatter.string(fromByteCount: bytes)
    }
}
