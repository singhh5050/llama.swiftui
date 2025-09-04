import Foundation

struct ModelStopTokens {
    static let stopTokenMappings: [String: [String]] = [
        // Meta Llama 3.x Series (Updated format)
        "llama-3": ["<|eot_id|>", "<|end_of_text|>"],
        "llama-3.1": ["<|eot_id|>", "<|end_of_text|>"],
        "llama-3.2": ["<|eot_id|>", "<|end_of_text|>"],
        
        // Legacy Llama 2.x Series
        "llama-2": ["</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
        "llama": ["</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
        "meta": ["</s>", "[INST]", "[/INST]", "<<SYS>>", "<</SYS>>"],
        
        // Microsoft Phi Series
        "phi": ["<|im_end|>", "<|endoftext|>"],
        "phi-4": ["<|im_end|>", "<|endoftext|>"],
        "microsoft": ["<|im_end|>", "<|endoftext|>"],
        
        // Qwen Series (Both 2.5 and 3.x use same format)
        "qwen": ["<|im_end|>", "<|endoftext|>"],
        "qwen2.5": ["<|im_end|>", "<|endoftext|>"],
        "qwen3": ["<|im_end|>", "<|endoftext|>"],
        
        // Google Gemma Series
        "gemma": ["<end_of_turn>"],
        "gemma-2": ["<end_of_turn>"],
        "gemma-3": ["<end_of_turn>"],
        "gemma-3n": ["<end_of_turn>"],
        "google": ["<end_of_turn>"],
        
        // Mistral AI
        "mistral": ["</s>", "[INST]", "[/INST]"],
        "mixtral": ["</s>", "[INST]", "[/INST]"],
        
        // OpenAI/ChatGPT style
        "gpt": ["<|im_end|>", "<|endoftext|>"],
        "openai": ["<|im_end|>", "<|endoftext|>"],
        
        // Anthropic Claude
        "claude": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"],
        "anthropic": ["Human:", "Assistant:", "\n\nHuman:", "\n\nAssistant:"],
        
        // Cohere
        "command": ["<|END_OF_TURN_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>"],
        "cohere": ["<|END_OF_TURN_TOKEN|>", "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>"],
        
        // Other common formats
        "alpaca": ["### Input:", "### Response:", "### Instruction:"],
        "vicuna": ["USER:", "ASSISTANT:", "SYSTEM:"],
        "wizard": ["### Instruction:", "### Response:"],
        "orca": ["<|im_start|>", "<|im_end|>"],
        
        // Default fallback
        "default": ["</s>", "<|endoftext|>", "\n\n"]
    ]
    
    static func getStopTokens(for modelName: String) -> [String] {
        let lowercaseName = modelName.lowercased()
        
        // Try to find matching company/format
        for (key, tokens) in stopTokenMappings {
            if lowercaseName.contains(key) {
                return tokens
            }
        }
        
        // Fallback to default
        return stopTokenMappings["default"] ?? ["</s>"]
    }
    
    static func detectModelCompany(from filename: String) -> String {
        let name = filename.lowercased()
        print("🔍 Model detection for filename: '\(filename)' -> lowercase: '\(name)'")
        
        // Specific model version detection (most specific first)
        if name.contains("llama-3.2") || name.contains("llama_3.2") {
            print("✅ Detected as llama-3.2")
            return "llama-3.2"
        } else if name.contains("llama-3.1") || name.contains("llama_3.1") {
            return "llama-3.1"
        } else if name.contains("llama-3") || name.contains("llama_3") {
            return "llama-3"
        } else if name.contains("llama-2") || name.contains("llama_2") {
            return "llama-2"
        } else if name.contains("llama") || name.contains("meta") {
            return "llama"
        }
        
        // Microsoft Phi detection
        else if name.contains("phi-4") || name.contains("phi_4") {
            return "phi-4"
        } else if name.contains("phi") || name.contains("microsoft") {
            return "phi"
        }
        
        // Qwen detection
        else if name.contains("qwen3") || name.contains("qwen_3") {
            return "qwen3"
        } else if name.contains("qwen2.5") || name.contains("qwen_2.5") {
            return "qwen2.5"
        } else if name.contains("qwen") {
            return "qwen"
        }
        
        // Google Gemma detection
        else if name.contains("gemma-3n") || name.contains("gemma_3n") {
            return "gemma-3n"
        } else if name.contains("gemma-3") || name.contains("gemma_3") {
            return "gemma-3"
        } else if name.contains("gemma-2") || name.contains("gemma_2") {
            return "gemma-2"
        } else if name.contains("gemma") || name.contains("google") {
            print("⚠️ Detected as gemma (this might be wrong!)")
            return "gemma"
        }
        
        // Other models
        else if name.contains("mistral") || name.contains("mixtral") {
            return "mistral"
        } else if name.contains("claude") || name.contains("anthropic") {
            return "anthropic"
        } else if name.contains("gpt") || name.contains("openai") {
            return "openai"
        } else if name.contains("command") || name.contains("cohere") {
            return "cohere"
        } else if name.contains("alpaca") {
            return "alpaca"
        } else if name.contains("vicuna") {
            return "vicuna"
        } else if name.contains("wizard") {
            return "wizard"
        } else if name.contains("orca") {
            return "orca"
        }
        
        print("⚠️ Using default detection")
        return "default"
    }
    
    // MARK: - Prompt Templates
    static func getPromptTemplate(for modelType: String) -> String {
        switch modelType {
        case "llama-3", "llama-3.1", "llama-3.2":
            return """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{SYSTEM_PROMPT}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{USER_MESSAGE}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            
        case "phi", "phi-4", "microsoft":
            return """
<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{USER_MESSAGE}<|im_end|>
<|im_start|>assistant
"""
            
        case "qwen", "qwen2.5", "qwen3":
            return """
<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
{USER_MESSAGE}<|im_end|>
<|im_start|>assistant
"""
            
        case "gemma", "gemma-2", "gemma-3", "gemma-3n", "google":
            return """
<start_of_turn>system
{SYSTEM_PROMPT}<end_of_turn>
<start_of_turn>user
{USER_MESSAGE}<end_of_turn>
<start_of_turn>model
"""
            
        case "llama-2", "llama":
            return """
<s>[INST] <<SYS>>
{SYSTEM_PROMPT}
<</SYS>>

{USER_MESSAGE} [/INST]
"""
            
        case "mistral", "mixtral":
            return """
<s>[INST] {SYSTEM_PROMPT}

{USER_MESSAGE} [/INST]
"""
            
        default:
            return """
### System:
{SYSTEM_PROMPT}

### User:
{USER_MESSAGE}

### Assistant:
"""
        }
    }
    
    static func getModelInfo(for filename: String) -> (type: String, stopTokens: [String], promptTemplate: String) {
        let detectedType = detectModelCompany(from: filename)
        let stopTokens = getStopTokens(for: detectedType)
        let promptTemplate = getPromptTemplate(for: detectedType)
        
        print("🎯 Final model info - Type: '\(detectedType)', Stop tokens: \(stopTokens)")
        
        return (type: detectedType, stopTokens: stopTokens, promptTemplate: promptTemplate)
    }
    
    // MARK: - Default System Prompt
    static let defaultSystemPrompt = """
You are a clear and focused assistant.

Aim for concise, complete answers.

Use plain language and short paragraphs.

When listing points, use brief bullet points or numbers.

Wrap up naturally once the question is fully addressed.

Keep the tone helpful and conversational.
"""
    
    static func formatPrompt(template: String, systemPrompt: String? = nil, userMessage: String) -> String {
        let system = systemPrompt ?? defaultSystemPrompt
        return template
            .replacingOccurrences(of: "{SYSTEM_PROMPT}", with: system)
            .replacingOccurrences(of: "{USER_MESSAGE}", with: userMessage)
    }
}
