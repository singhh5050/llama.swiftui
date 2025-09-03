import Foundation
import llama

enum LlamaError: Error {
    case couldNotInitializeContext
}

func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

func llama_batch_add(_ batch: inout llama_batch, 
                     _ id: llama_token, 
                     _ pos: llama_pos, 
                     _ seq_ids: [llama_seq_id], 
                     _ logits: Bool, 
                     capacity: Int, 
                     nSeqMax: Int32) {
    let idx = Int(batch.n_tokens)
    precondition(idx < capacity, "llama_batch_add: capacity exceeded (\(idx) >= \(capacity))")
    precondition(seq_ids.count <= Int(nSeqMax), "llama_batch_add: seq_ids.count (\(seq_ids.count)) > nSeqMax (\(nSeqMax))")

    batch.token[idx]    = id
    batch.pos[idx]      = pos
    batch.n_seq_id[idx] = Int32(seq_ids.count)

    guard let seqBuf = batch.seq_id[idx] else {
        fatalError("llama_batch_add: seq_id[\(idx)] is nil — bad batch init or ABI mismatch")
    }
    for i in 0..<seq_ids.count { seqBuf[i] = seq_ids[i] }

    batch.logits[idx] = logits ? 1 : 0
    batch.n_tokens += 1
}

actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var tokens_list: [llama_token]
    var is_done: Bool = false
    
    // Track batch capacity for safe allocation
    private var batchCapacity: Int = 256
    private var batchSeqMax: Int32 = 1

    /// This variable is used to store temporarily invalid cchars
    private var temporary_invalid_cchars: [CChar]
    
    /// Stop tokens for this model
    private var stopTokens: [String] = []
    private var generatedText: String = ""

    var n_len: Int32 = 1024
    var n_cur: Int32 = 0

    var n_decode: Int32 = 0
    
    // Benchmarking metrics
    private var prefillStartTime: UInt64 = 0
    private var firstTokenTime: UInt64 = 0
    private var decodeStartTime: UInt64 = 0
    private var totalPrefillTokens: Int = 0
    private var totalDecodeTokens: Int = 0
    private var isFirstToken: Bool = true

    init(model: OpaquePointer, context: OpaquePointer, stopTokens: [String] = []) {
        self.model = model
        self.context = context
        self.tokens_list = []
        self.batchCapacity = 256
        self.batchSeqMax = 1
        self.batch = llama_batch_init(Int32(batchCapacity), 0, batchSeqMax)
        
        // Sanity check - catches token/embedding mode mismatch instantly
        precondition(batch.token != nil, "token buffer is nil; did you init llama_batch with embd > 0?")
        
        self.temporary_invalid_cchars = []
        self.stopTokens = stopTokens
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.4))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        vocab = llama_model_get_vocab(model)
    }

    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_model_free(model)
        llama_free(context)
        llama_backend_free()
    }
    
    // Helper to ensure batch capacity
    private func ensureBatchCapacity(_ required: Int, nSeqMax: Int32 = 1) {
        if required <= batchCapacity && nSeqMax <= batchSeqMax { return }
        // grow to max(required, oldCapacity) and max seq lanes
        batchCapacity = max(required, batchCapacity)
        batchSeqMax   = max(nSeqMax, batchSeqMax)
        llama_batch_free(batch)
        batch = llama_batch_init(Int32(batchCapacity), 0, batchSeqMax)
        
        // Sanity check - catches token/embedding mode mismatch instantly
        precondition(batch.token != nil, "token buffer is nil; did you init llama_batch with embd > 0?")
    }

    static func create_context(path: String, stopTokens: [String] = []) async throws -> LlamaContext {
        llama_backend_init()
        var model_params = llama_model_default_params()

#if targetEnvironment(simulator)
        model_params.n_gpu_layers = 0
        print("Running on simulator, force use n_gpu_layers = 0")
#else
        // Use maximum GPU layers for performance (high number ensures all layers are used)
        model_params.n_gpu_layers = 99  // High number to ensure all model layers are offloaded
        print("Using maximum GPU layers (99) for optimal performance")
#endif
        let model = llama_model_load_from_file(path, model_params)
        guard let model else {
            print("Could not load model at \(path)")
            throw LlamaError.couldNotInitializeContext
        }

        let n_threads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        print("Using \(n_threads) threads")

        var ctx_params = llama_context_default_params()
        
        // Standardized context and batch settings for benchmarking
        ctx_params.n_ctx = 1024
        ctx_params.n_batch = 256
        ctx_params.n_threads = Int32(n_threads)
        ctx_params.n_threads_batch = Int32(n_threads)

        let context = llama_init_from_model(model, ctx_params)
        guard let context else {
            print("Could not load context!")
            throw LlamaError.couldNotInitializeContext
        }

        let llamaContext = LlamaContext(model: model, context: context, stopTokens: stopTokens)
        return llamaContext
    }

    func model_info() -> String {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 256)
        result.initialize(repeating: Int8(0), count: 256)
        defer {
            result.deallocate()
        }

        // TODO: this is probably very stupid way to get the string from C

        let nChars = llama_model_desc(model, result, 256)
        let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nChars))

        var SwiftString = ""
        for char in bufferPointer {
            SwiftString.append(Character(UnicodeScalar(UInt8(char))))
        }

        return SwiftString
    }

    func get_n_tokens() -> Int32 {
        return batch.n_tokens;
    }

    // Helpers for chat template
    private func buildSystemPrefix(_ system: String) -> String {
        """
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        \(system)
        <|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        """
    }
    
    private func buildUserSuffix(_ user: String) -> String {
        """
        \(user)
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """
    }
    
    /// Feed full system + trimmed user, then run prefill.
    /// - maxNewTokens: tokens to generate (sets n_len)
    /// - maxUserTokens: hard cap on user tokens before fitting to n_ctx
    func completion_init(system: String,
                         user: String,
                         maxNewTokens: Int = 512,
                         maxUserTokens: Int = .max) {
        print("attempting to complete (system len=\(system.count), user len=\(user.count))")

        // Set generation length based on user request
        n_len = Int32(max(1, maxNewTokens))

        // Reset benchmarking metrics
        prefillStartTime = DispatchTime.now().uptimeNanoseconds
        firstTokenTime = 0
        decodeStartTime = 0
        totalDecodeTokens = 0
        isFirstToken = true
        temporary_invalid_cchars = []
        generatedText = ""

        // Build strings for tokenization
        let sysPrefix = buildSystemPrefix(system)
        let userBlock = buildUserSuffix(user)

        // Tokenize separately so we can protect system
        let sysTok  = tokenize(text: sysPrefix, add_bos: true)
        var userTok = tokenize(text: userBlock, add_bos: false)

        // Optional hard cap on user segment (keeps the *tail* / most recent)
        if userTok.count > maxUserTokens {
            userTok = Array(userTok.suffix(maxUserTokens))
            print("User trimmed to maxUserTokens=\(maxUserTokens)")
        }

        // Fit to context: keep all system tokens, trim user only
        let nctx = Int(llama_n_ctx(context))
        let reserve = min(maxNewTokens, nctx - 1)        // room for generation
        let budgetForPrompt = max(1, nctx - reserve)     // room for system+user

        var availForUser = max(0, budgetForPrompt - sysTok.count)
        if availForUser < userTok.count {
            userTok = Array(userTok.suffix(availForUser))
            print("User trimmed to fit n_ctx: \(userTok.count) tokens (reserved \(reserve))")
        }

        // Final prompt tokens (system kept fully)
        let toks = sysTok + userTok
        totalPrefillTokens = toks.count
        tokens_list = toks

        // Ensure batch capacity can hold the entire prefill in one shot
        ensureBatchCapacity(toks.count, nSeqMax: 1)

        print("\n n_len = \(n_len), n_ctx = \(nctx), prefill_tokens = \(toks.count), reserve = \(reserve)")

        for id in toks {
            print(String(cString: token_to_piece(token: id) + [0]))
        }

        llama_batch_clear(&batch)

        // Add all prefill tokens
        for (i, id) in toks.enumerated() {
            llama_batch_add(&batch, id, Int32(i), [0], false, capacity: batchCapacity, nSeqMax: batchSeqMax)
        }
        // Request logits for the last prefill token
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        // Decode prefill
        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed")
        }

        n_cur = batch.n_tokens
        
        // Mark end of prefill phase
        decodeStartTime = DispatchTime.now().uptimeNanoseconds
    }
    
    func completion_init(text: String) {
        print("attempting to complete \"\(text)\"")

        // Reset benchmarking metrics
        prefillStartTime = DispatchTime.now().uptimeNanoseconds
        firstTokenTime = 0
        decodeStartTime = 0
        totalDecodeTokens = 0
        isFirstToken = true

        var toks = tokenize(text: text, add_bos: true)
        totalPrefillTokens = toks.count
        temporary_invalid_cchars = []
        generatedText = ""

        let n_ctx = Int(llama_n_ctx(context))

        // Reserve space for generation - max prefill tokens
        let maxPrompt = max(1, n_ctx - Int(n_len))

        if toks.count > maxPrompt {
            // keep the most recent tokens (tail) so semantics stay freshest
            toks = Array(toks.suffix(maxPrompt))
            print("Prefill truncated to \(toks.count) to fit n_ctx \(n_ctx) with \(n_len) reserved")
        }

        // Ensure batch capacity can hold the entire prefill in one shot
        ensureBatchCapacity(toks.count, nSeqMax: 1)

        print("\n n_len = \(n_len), n_ctx = \(n_ctx), prefill_tokens = \(toks.count)")

        for id in toks {
            print(String(cString: token_to_piece(token: id) + [0]))
        }

        llama_batch_clear(&batch)

        // Add all prefill tokens
        for (i, id) in toks.enumerated() {
            llama_batch_add(&batch, id, Int32(i), [0], false, capacity: batchCapacity, nSeqMax: batchSeqMax)
        }
        // Request logits for the last prefill token
        batch.logits[Int(batch.n_tokens) - 1] = 1 // true

        // Decode prefill
        if llama_decode(context, batch) != 0 {
            print("llama_decode() failed")
        }

        n_cur = batch.n_tokens
        tokens_list = toks // keep if you want to track them
        
        // Mark end of prefill phase
        decodeStartTime = DispatchTime.now().uptimeNanoseconds
    }

    func completion_loop() -> String {
        // Capture first token time
        if isFirstToken {
            firstTokenTime = DispatchTime.now().uptimeNanoseconds
            isFirstToken = false
        }
        
        var new_token_id: llama_token = 0

        new_token_id = llama_sampler_sample(sampling, context, batch.n_tokens - 1)

        // Hard stop if context is full
        if n_cur >= Int32(llama_n_ctx(context)) {
            print("\nStopped: context full")
            is_done = true
            let new_token_str = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            return new_token_str
        }
        
        if llama_vocab_is_eog(vocab, new_token_id) || n_cur == n_len {
            print("\n")
            is_done = true
            let new_token_str = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            return new_token_str
        }

        let new_token_cchars = token_to_piece(token: new_token_id)
        temporary_invalid_cchars.append(contentsOf: new_token_cchars)
        let new_token_str: String
        if let string = String(validatingUTF8: temporary_invalid_cchars + [0]) {
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else if (0 ..< temporary_invalid_cchars.count).contains(where: {$0 != 0 && String(validatingUTF8: Array(temporary_invalid_cchars.suffix($0)) + [0]) != nil}) {
            // in this case, at least the suffix of the temporary_invalid_cchars can be interpreted as UTF8 string
            let string = String(cString: temporary_invalid_cchars + [0])
            temporary_invalid_cchars.removeAll()
            new_token_str = string
        } else {
            new_token_str = ""
        }
        
        // Count decode tokens
        totalDecodeTokens += 1
        
        // Check for stop tokens
        generatedText += new_token_str
        for stopToken in stopTokens {
            if generatedText.contains(stopToken) {
                print("\nStopped due to stop token: \(stopToken)")
                is_done = true
                // Return text up to the stop token
                if let range = generatedText.range(of: stopToken) {
                    let finalText = String(generatedText[..<range.lowerBound])
                    return finalText.replacingOccurrences(of: generatedText, with: new_token_str)
                }
                return new_token_str
            }
        }
        
        print(new_token_str)
        // tokens_list.append(new_token_id)

        llama_batch_clear(&batch)
        // only 1 token per decode step → capacity 1 is enough, but we'll reuse the same batch
        ensureBatchCapacity(1, nSeqMax: 1)
        llama_batch_add(&batch, new_token_id, n_cur, [0], true, capacity: batchCapacity, nSeqMax: batchSeqMax)

        n_decode += 1
        n_cur    += 1

        if llama_decode(context, batch) != 0 {
            print("failed to evaluate llama!")
        }

        return new_token_str
    }

    func bench(pp: Int, tg: Int, pl: Int, nr: Int = 1) -> String {
        var pp_avg: Double = 0
        var tg_avg: Double = 0

        var pp_std: Double = 0
        var tg_std: Double = 0

        for _ in 0..<nr {
            // bench prompt processing

            llama_batch_clear(&batch)

            let n_tokens = pp
            
            // Ensure batch capacity for benchmark
            ensureBatchCapacity(n_tokens, nSeqMax: 1)

            for i in 0..<n_tokens {
                llama_batch_add(&batch, 0, Int32(i), [0], false, capacity: batchCapacity, nSeqMax: batchSeqMax)
            }
            batch.logits[Int(batch.n_tokens) - 1] = 1 // true

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp_start = DispatchTime.now().uptimeNanoseconds / 1000;

            if llama_decode(context, batch) != 0 {
                print("llama_decode() failed during prompt")
            }
            llama_synchronize(context)

            let t_pp_end = DispatchTime.now().uptimeNanoseconds / 1000;

            // bench text generation

            llama_memory_clear(llama_get_memory(context), false)

            let t_tg_start = DispatchTime.now().uptimeNanoseconds / 1000;

            for i in 0..<tg {
                llama_batch_clear(&batch)
                
                // Ensure batch capacity for parallel sequences
                ensureBatchCapacity(pl, nSeqMax: Int32(pl))

                for j in 0..<pl {
                    llama_batch_add(&batch, 0, Int32(i), [Int32(j)], true, capacity: batchCapacity, nSeqMax: batchSeqMax)
                }

                if llama_decode(context, batch) != 0 {
                    print("llama_decode() failed during text generation")
                }
                llama_synchronize(context)
            }

            let t_tg_end = DispatchTime.now().uptimeNanoseconds / 1000;

            llama_memory_clear(llama_get_memory(context), false)

            let t_pp = Double(t_pp_end - t_pp_start) / 1000000.0
            let t_tg = Double(t_tg_end - t_tg_start) / 1000000.0

            let speed_pp = Double(pp)    / t_pp
            let speed_tg = Double(pl*tg) / t_tg

            pp_avg += speed_pp
            tg_avg += speed_tg

            pp_std += speed_pp * speed_pp
            tg_std += speed_tg * speed_tg

            print("pp \(speed_pp) t/s, tg \(speed_tg) t/s")
        }

        pp_avg /= Double(nr)
        tg_avg /= Double(nr)

        if nr > 1 {
            pp_std = sqrt(pp_std / Double(nr - 1) - pp_avg * pp_avg * Double(nr) / Double(nr - 1))
            tg_std = sqrt(tg_std / Double(nr - 1) - tg_avg * tg_avg * Double(nr) / Double(nr - 1))
        } else {
            pp_std = 0
            tg_std = 0
        }

        let model_desc     = model_info();
        let model_size     = String(format: "%.2f GiB", Double(llama_model_size(model)) / 1024.0 / 1024.0 / 1024.0);
        let model_n_params = String(format: "%.2f B", Double(llama_model_n_params(model)) / 1e9);
        let backend        = "Metal";
        let pp_avg_str     = String(format: "%.2f", pp_avg);
        let tg_avg_str     = String(format: "%.2f", tg_avg);
        let pp_std_str     = String(format: "%.2f", pp_std);
        let tg_std_str     = String(format: "%.2f", tg_std);

        var result = ""

        result += String("| model | size | params | backend | test | t/s |\n")
        result += String("| --- | --- | --- | --- | --- | --- |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | pp \(pp) | \(pp_avg_str) ± \(pp_std_str) |\n")
        result += String("| \(model_desc) | \(model_size) | \(model_n_params) | \(backend) | tg \(tg) | \(tg_avg_str) ± \(tg_std_str) |\n")

        return result;
    }

    func clear() {
        tokens_list.removeAll()
        temporary_invalid_cchars.removeAll()
        generatedText.removeAll()
        llama_memory_clear(llama_get_memory(context), true)
    }
    

    
    func getBenchmarkMetrics() -> (timeToFirstToken: Double, prefillLatency: Double, decodeLatency: Double, prefillTokens: Int, decodeTokens: Int) {
        let currentTime = DispatchTime.now().uptimeNanoseconds
        let NS_PER_MS = 1_000_000.0
        
        let ttft = firstTokenTime > 0 ? Double(firstTokenTime - prefillStartTime) / NS_PER_MS : 0.0
        let prefillTime = decodeStartTime > 0 ? Double(decodeStartTime - prefillStartTime) / NS_PER_MS : 0.0
        let decodeTime = firstTokenTime > 0 ? Double(currentTime - firstTokenTime) / NS_PER_MS : 0.0
        
        return (
            timeToFirstToken: ttft,
            prefillLatency: prefillTime,
            decodeLatency: decodeTime,
            prefillTokens: totalPrefillTokens,
            decodeTokens: totalDecodeTokens
        )
    }
    
    func model_size() -> Int64 {
        return Int64(llama_model_size(model))
    }
    
    func model_params() -> Int64 {
        return Int64(llama_model_n_params(model))
    }

    private func tokenize(text: String, add_bos: Bool) -> [llama_token] {
        let utf8Count = text.utf8.count
        let n_tokens = utf8Count + (add_bos ? 1 : 0) + 1
        let tokens = UnsafeMutablePointer<llama_token>.allocate(capacity: n_tokens)
        let tokenCount = llama_tokenize(vocab, text, Int32(utf8Count), tokens, Int32(n_tokens), add_bos, false)

        var swiftTokens: [llama_token] = []
        for i in 0..<tokenCount {
            swiftTokens.append(tokens[Int(i)])
        }

        tokens.deallocate()

        return swiftTokens
    }

    /// - note: The result does not contain null-terminator
    private func token_to_piece(token: llama_token) -> [CChar] {
        let result = UnsafeMutablePointer<Int8>.allocate(capacity: 8)
        result.initialize(repeating: Int8(0), count: 8)
        defer {
            result.deallocate()
        }
        let nTokens = llama_token_to_piece(vocab, token, result, 8, 0, false)

        if nTokens < 0 {
            let newResult = UnsafeMutablePointer<Int8>.allocate(capacity: Int(-nTokens))
            newResult.initialize(repeating: Int8(0), count: Int(-nTokens))
            defer {
                newResult.deallocate()
            }
            let nNewTokens = llama_token_to_piece(vocab, token, newResult, -nTokens, 0, false)
            let bufferPointer = UnsafeBufferPointer(start: newResult, count: Int(nNewTokens))
            return Array(bufferPointer)
        } else {
            let bufferPointer = UnsafeBufferPointer(start: result, count: Int(nTokens))
            return Array(bufferPointer)
        }
    }
}
