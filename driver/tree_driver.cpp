#include "llama.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

using json = nlohmann::json;

struct Node {
    int id, parent;
    std::vector<llama_token> tokens;
    std::vector<llama_pos> pos;
    std::vector<llama_seq_id> seqs;
    std::vector<int> children;
};

static void fail(const std::string & msg) {
    fprintf(stderr, "%s\n", msg.c_str());
    exit(1);
}

static std::vector<Node> load_doc(const std::string & path, int & n_seq) {
    std::ifstream f(path);
    if (!f) fail("cannot open " + path);
    json j = json::parse(f);
    n_seq = j["n_seq"];
    std::vector<Node> nodes;
    for (auto & jn : j["nodes"]) {
        Node n;
        n.id = jn["id"];
        n.parent = jn["parent"];
        n.tokens = jn["tokens"].get<std::vector<llama_token>>();
        n.pos = jn["positions"].get<std::vector<llama_pos>>();
        n.seqs = jn["seq_ids"].get<std::vector<llama_seq_id>>();
        nodes.push_back(n);
    }
    for (auto & n : nodes) {
        if (n.parent >= 0) nodes[n.parent].children.push_back(n.id);
    }
    return nodes;
}

static double max_abs_diff(const std::vector<float> & a, const std::vector<float> & b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) m = std::max(m, (double) std::fabs(a[i] - b[i]));
    return m;
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void batch_add(llama_batch & b, llama_token t, llama_pos p, const std::vector<llama_seq_id> & seqs, bool out) {
    const int i = b.n_tokens++;
    b.token[i] = t;
    b.pos[i] = p;
    b.n_seq_id[i] = (int32_t) seqs.size();
    for (size_t s = 0; s < seqs.size(); ++s) b.seq_id[i][s] = seqs[s];
    b.logits[i] = out;
}

static void decode(llama_context * ctx, llama_batch & b) {
    const int ret = llama_decode(ctx, b);
    if (ret != 0) fail("llama_decode failed: " + std::to_string(ret));
}

static std::vector<float> logits_at(llama_context * ctx, const llama_vocab * vocab, int idx) {
    const float * lg = llama_get_logits_ith(ctx, idx);
    return std::vector<float>(lg, lg + llama_vocab_n_tokens(vocab));
}

static llama_token greedy(llama_context * ctx, const llama_vocab * vocab, int idx) {
    const float * lg = llama_get_logits_ith(ctx, idx);
    return (llama_token) (std::max_element(lg, lg + llama_vocab_n_tokens(vocab)) - lg);
}

static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int n = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, false, false);
    std::vector<llama_token> out(std::max(n, 0));
    if (n > 0) llama_tokenize(vocab, text.c_str(), text.size(), out.data(), n, false, false);
    return out;
}

static std::string detok(const llama_vocab * vocab, const std::vector<llama_token> & tokens) {
    std::string s;
    char buf[256];
    for (llama_token t : tokens) {
        const int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, false);
        if (n > 0) s.append(buf, n);
    }
    return s;
}

static void prefill(llama_context * ctx, const std::vector<const Node *> & nodes, int n_seq_max) {
    int n = 0;
    for (auto * node : nodes) n += node->tokens.size();
    if (n == 0) return;
    llama_batch b = llama_batch_init(n, 0, n_seq_max);
    for (auto * node : nodes) {
        for (size_t k = 0; k < node->tokens.size(); ++k) batch_add(b, node->tokens[k], node->pos[k], node->seqs, false);
    }
    decode(ctx, b);
    llama_batch_free(b);
}

static void collect(const std::vector<Node> & nodes, int id, std::vector<const Node *> & out) {
    out.push_back(&nodes[id]);
    for (int c : nodes[id].children) collect(nodes, c, out);
}

static std::string ask(llama_context * ctx, const llama_vocab * vocab, const Node & target, llama_seq_id scratch,
                       const std::string & question, int n_predict, std::vector<float> & first) {
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, scratch, -1, -1);
    llama_memory_seq_cp(mem, target.seqs[0], scratch, -1, -1);
    llama_pos p = llama_memory_seq_pos_max(mem, scratch) + 1;

    std::vector<llama_token> q = tokenize(vocab, "\nQuestion: " + question + "\nAnswer:");
    llama_batch b = llama_batch_init(q.size(), 0, 1);
    for (size_t i = 0; i < q.size(); ++i) batch_add(b, q[i], p++, {scratch}, i + 1 == q.size());
    decode(ctx, b);
    first = logits_at(ctx, vocab, -1);

    std::vector<llama_token> answer;
    for (int k = 0; k < n_predict; ++k) {
        const llama_token t = greedy(ctx, vocab, -1);
        if (llama_vocab_is_eog(vocab, t)) break;
        const bool newline = detok(vocab, {t}).find('\n') != std::string::npos;
        if (newline && detok(vocab, answer).find_first_not_of(" \t\n") != std::string::npos) break;
        answer.push_back(t);
        b.n_tokens = 0;
        batch_add(b, t, p++, {scratch}, true);
        decode(ctx, b);
    }
    llama_batch_free(b);
    llama_memory_seq_rm(mem, scratch, -1, -1);
    return detok(vocab, answer);
}

static int evict(llama_context * ctx, const std::vector<Node> & nodes, int id, std::pair<llama_pos, llama_pos> & range) {
    std::vector<const Node *> sub;
    collect(nodes, id, sub);
    llama_pos p0 = nodes[id].pos.empty() ? 0 : nodes[id].pos.front();
    llama_pos p1 = p0;
    int n_tokens = 0;
    for (auto * node : sub) {
        n_tokens += node->tokens.size();
        if (!node->pos.empty()) p1 = std::max(p1, node->pos.back() + 1);
    }
    llama_memory_t mem = llama_get_memory(ctx);
    for (llama_seq_id s : nodes[id].seqs) llama_memory_seq_rm(mem, s, p0, p1);
    range = {p0, p1};
    return n_tokens;
}

static void edit_leaf(llama_context * ctx, const llama_vocab * vocab, std::vector<Node> & nodes, int id,
                      const std::string & text, int n_seq_max) {
    Node & node = nodes[id];
    if (!node.children.empty()) fail("--edit expects a leaf node");
    const llama_pos p0 = node.pos.empty() ? 0 : node.pos.front();
    std::pair<llama_pos, llama_pos> range;
    evict(ctx, nodes, id, range);
    node.tokens = tokenize(vocab, text);
    node.pos.clear();
    for (size_t k = 0; k < node.tokens.size(); ++k) node.pos.push_back(p0 + (llama_pos) k);
    prefill(ctx, {&node}, n_seq_max);
}

static std::vector<std::string> generate(llama_context * ctx, const llama_vocab * vocab,
                                         const std::vector<std::string> & prompts, int n_predict,
                                         std::vector<std::vector<float>> & first, std::vector<int> & lengths) {
    const int n = prompts.size();
    llama_memory_clear(llama_get_memory(ctx), true);
    std::vector<std::vector<llama_token>> inputs;
    int total = 0;
    for (auto & prompt : prompts) {
        std::vector<llama_token> t = tokenize(vocab, prompt);
        t.insert(t.begin(), llama_vocab_bos(vocab));
        total += t.size();
        inputs.push_back(t);
    }
    llama_batch b = llama_batch_init(std::max(total, n), 0, 1);
    std::vector<llama_pos> p(n, 0);
    std::vector<int> last(n);
    for (int s = 0; s < n; ++s) {
        for (size_t i = 0; i < inputs[s].size(); ++i) batch_add(b, inputs[s][i], p[s]++, {s}, i + 1 == inputs[s].size());
        last[s] = b.n_tokens - 1;
        lengths.push_back(inputs[s].size());
    }
    decode(ctx, b);
    for (int s = 0; s < n; ++s) first.push_back(logits_at(ctx, vocab, last[s]));
    std::vector<std::vector<llama_token>> out(n);
    for (int k = 0; k < n_predict; ++k) {
        std::vector<llama_token> next(n);
        for (int s = 0; s < n; ++s) next[s] = greedy(ctx, vocab, last[s]);
        b.n_tokens = 0;
        for (int s = 0; s < n; ++s) {
            out[s].push_back(next[s]);
            last[s] = b.n_tokens;
            batch_add(b, next[s], p[s]++, {s}, true);
        }
        decode(ctx, b);
    }
    llama_batch_free(b);
    std::vector<std::string> texts;
    for (auto & o : out) texts.push_back(detok(vocab, o));
    return texts;
}

static json test_two_prompts(llama_context * ctx, const llama_vocab * vocab, const std::string & p1,
                             const std::string & p2, int n_predict) {
    std::vector<std::vector<float>> lt, l1, l2, lx;
    std::vector<int> lengths, unused;
    auto both = generate(ctx, vocab, {p1, p2}, n_predict, lt, lengths);
    auto a1 = generate(ctx, vocab, {p1}, n_predict, l1, unused);
    auto a2 = generate(ctx, vocab, {p2}, n_predict, l2, unused);
    generate(ctx, vocab, {p1, "Completely different text in the other sequence."}, 1, lx, unused);
    return {{"together", both}, {"alone", {a1[0], a2[0]}}, {"match", both[0] == a1[0] && both[1] == a2[0]},
            {"prompt_tokens", lengths},
            {"first_logit_diff_together_vs_alone", {max_abs_diff(lt[0], l1[0]), max_abs_diff(lt[1], l2[0])}},
            {"seq0_logit_diff_when_seq1_text_changes", max_abs_diff(lt[0], lx[0])}};
}

static void fill_holes(llama_context * ctx, const llama_vocab * vocab, llama_seq_id seq, int n) {
    if (n <= 0) return;
    llama_batch b = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) batch_add(b, llama_vocab_bos(vocab), i, {seq}, false);
    decode(ctx, b);
    llama_batch_free(b);
}

static json test_tree(llama_context * ctx, const llama_vocab * vocab, std::vector<Node> & nodes, int n_seq_max,
                      int ask_node, const std::string & question, int edit_node, const std::string & edit_text,
                      int evict_node, int n_predict, bool keep_layout) {
    const llama_seq_id scratch = n_seq_max - 1;
    json out;
    std::vector<const Node *> all;
    for (auto & n : nodes) all.push_back(&n);
    double t0 = now_ms();
    prefill(ctx, all, n_seq_max);
    out["prefill_ms"] = now_ms() - t0;
    out["kv_state_bytes"] = llama_state_get_size(ctx);

    std::vector<float> before, after;
    t0 = now_ms();
    out["answer_before"] = ask(ctx, vocab, nodes[ask_node], scratch, question, n_predict, before);
    out["ask_ms"] = now_ms() - t0;

    if (edit_node >= 0) {
        t0 = now_ms();
        edit_leaf(ctx, vocab, nodes, edit_node, edit_text, n_seq_max);
        out["edit_ms"] = now_ms() - t0;
        out["answer_after_edit"] = ask(ctx, vocab, nodes[ask_node], scratch, question, n_predict, after);
        out["max_logit_diff_edit"] = max_abs_diff(before, after);
    }
    if (evict_node >= 0) {
        const size_t bytes_before = llama_state_get_size(ctx);
        std::pair<llama_pos, llama_pos> range;
        t0 = now_ms();
        const int n_tokens = evict(ctx, nodes, evict_node, range);
        out["evict_ms"] = now_ms() - t0;
        out["evict_position_range"] = {range.first, range.second};
        out["evict_seq_ids"] = nodes[evict_node].seqs;
        out["evicted_tokens"] = n_tokens;
        out["kv_bytes_freed"] = (long long) bytes_before - (long long) llama_state_get_size(ctx);
        if (keep_layout) fill_holes(ctx, vocab, n_seq_max, n_tokens);
        out["answer_after_evict"] = ask(ctx, vocab, nodes[ask_node], scratch, question, n_predict, after);
        out["max_logit_diff_evict"] = max_abs_diff(before, after);
        if (keep_layout) llama_memory_seq_rm(llama_get_memory(ctx), n_seq_max, -1, -1);
    }
    return out;
}

int main(int argc, char ** argv) {
    const std::string usage = "usage: tree-driver -m MODEL -d DOC.json --ask NODE QUESTION [--edit NODE TEXT] "
                              "[--evict NODE] [-t THREADS] [-n N_PREDICT] [--keep-layout]\n"
                              "       tree-driver -m MODEL --two-prompts PROMPT1 PROMPT2 [-t THREADS] [-n N_PREDICT]";
    std::string model_path, doc_path, question, edit_text, p1, p2;
    int ask_node = -1, edit_node = -1, evict_node = -1, n_predict = 32, n_threads = 4;
    bool keep_layout = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--keep-layout") {
            keep_layout = true;
            continue;
        }
        const int need = (a == "--ask" || a == "--edit" || a == "--two-prompts") ? 2 : 1;
        if (i + need >= argc) fail(usage);
        if (a == "-m") model_path = argv[++i];
        else if (a == "-d") doc_path = argv[++i];
        else if (a == "-t") n_threads = atoi(argv[++i]);
        else if (a == "-n") n_predict = atoi(argv[++i]);
        else if (a == "--ask") { ask_node = atoi(argv[i + 1]); question = argv[i + 2]; i += 2; }
        else if (a == "--edit") { edit_node = atoi(argv[i + 1]); edit_text = argv[i + 2]; i += 2; }
        else if (a == "--evict") evict_node = atoi(argv[++i]);
        else if (a == "--two-prompts") { p1 = argv[i + 1]; p2 = argv[i + 2]; i += 2; }
        else fail(usage);
    }

    int n_seq = 2;
    std::vector<Node> nodes;
    int n_tokens = 0;
    if (p1.empty()) {
        if (doc_path.empty()) fail(usage);
        nodes = load_doc(doc_path, n_seq);
        const int n = nodes.size();
        if (ask_node < 0 || ask_node >= n || edit_node >= n || evict_node >= n) {
            fail("node ids must be in 0.." + std::to_string(n - 1));
        }
        for (auto & node : nodes) n_tokens += node.tokens.size();
    }
    if (model_path.empty()) fail(usage);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) fail("cannot load " + model_path);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = std::max(4096, n_tokens + 1024 + (int) edit_text.size());
    cp.n_batch = cp.n_ctx;
    cp.n_seq_max = n_seq + 2;
    cp.kv_unified = true;
    cp.n_threads = n_threads;
    cp.n_threads_batch = n_threads;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) fail("cannot create context");

    json out = p1.empty() ? test_tree(ctx, vocab, nodes, n_seq + 1, ask_node, question, edit_node, edit_text,
                                      evict_node, n_predict, keep_layout)
                          : test_two_prompts(ctx, vocab, p1, p2, n_predict);
    printf("%s\n", out.dump(1, ' ', false, json::error_handler_t::replace).c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
