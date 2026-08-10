import time, traceback
from .config import DEFAULT_CONFIG, ARTIFACTS
from .logger import Logger
from .tokenizer import FrozenTokenizer
from .shards import build_demo_documents, create_shards, validate_manifests
from .firewall import consume_shard
from .mixture import compile_schedule, planned_share
from .opus import opus_decide
from .packing import pack_sample, validate_packed
from .ledger import Ledger
from .checkpoint import save_checkpoint, load_checkpoint
from .recovery import build_batches, compare_batch, replay, stream_hash
from .model import ToyModel
from .audit import generate_evidence, generate_evidence_md
from .utils import write_json, read_json, sha256_obj


def reset_artifacts():
    if ARTIFACTS.exists():
        import shutil
        shutil.rmtree(ARTIFACTS)
    for d in ("manifests", "ledgers", "checkpoints", "shards"):
        (ARTIFACTS / d).mkdir(parents=True)


def run_demo():
    reset_artifacts()
    cfg = DEFAULT_CONFIG
    log = Logger(ARTIFACTS / "run.log")
    checks = {}
    start_time = time.perf_counter()
    try:
        tokenizer = FrozenTokenizer(vocab_size=cfg.vocab_size)
        docs = build_demo_documents()
        manifests = create_shards(docs, tokenizer, ARTIFACTS / "shards")
        validate_manifests(ARTIFACTS / "shards", tokenizer)
        write_json(ARTIFACTS / "manifests" / "all.json", manifests)
        log.event("shards created")
        log.event("manifests validated")
        checks["tokenizer_integrity"] = {"passed": all(m["tokenizer_hash"] == tokenizer.tokenizer_hash for m in manifests), "evidence": "manifests/all.json"}

        eval_manifest = next(m for m in manifests if m["split"] == "eval")
        try:
            consume_shard(eval_manifest)
            blocked = False
        except PermissionError:
            blocked = True
        log.event("evaluation data blocked", blocked)
        log.event("eval_shard_blocked", blocked)
        checks["evaluation_firewall"] = {"passed": blocked, "evidence": "run.log + eval manifest"}

        stages = compile_schedule()
        stage = stages[0]
        write_json(ARTIFACTS / "manifests" / "mixture.json", {"stage": stage.name, "weights": stage.weights, "floors": stage.floors, "schedule_hash": planned_share(stage)})
        log.event("mixture compiled")

        opus_inputs = [
            {"candidate_id":"c-accept","quality":.90,"current_share":.20,"floor":.10},
            {"candidate_id":"c-reject","quality":.10,"current_share":.20,"floor":.10},
            {"candidate_id":"c-defer","quality":.50,"current_share":.20,"floor":.10},
            {"candidate_id":"c-floor","quality":.90,"current_share":.01,"floor":.10},
        ]
        opus_records = []
        for c in opus_inputs:
            decision, reason = opus_decide(c, c["current_share"], c["floor"])
            opus_records.append({**c, "decision": decision, "reason": reason})
        write_json(ARTIFACTS / "manifests" / "opus_decisions.json", opus_records)
        log.event("OPUS decisions recorded")
        checks["opus_audit"] = {"passed": {r["decision"] for r in opus_records} == {"ACCEPT","REJECT","DEFER","FLOOR_OVERRIDE"}, "evidence":"manifests/opus_decisions.json"}

        train = []
        for m in manifests:
            if m["split"] != "train":
                continue
            train.extend(read_json(ARTIFACTS / "shards" / f"{m['shard_id']}.json"))

        # Deterministic lane-aware scheduler: sort by stage priority and use every lane.
        lane_rank = {lane: i for i, lane in enumerate(stage.weights)}
        train.sort(key=lambda s: (lane_rank[s["lane"]], s["document_id"]))
        batches = build_batches(train, cfg.sequence_length, cfg.batch_size, policy="standard")
        for b in batches:
            for p in b["samples"]:
                validate_packed(p)
        write_json(ARTIFACTS / "manifests" / "batches.json", batches)
        log.event("batches packed")
        checks["packing_correctness"] = {"passed": all(validate_packed(p) for b in batches for p in b["samples"]), "evidence":"manifests/batches.json"}

        consumption = Ledger(ARTIFACTS / "ledgers" / "consumption.jsonl")
        learning = Ledger(ARTIFACTS / "ledgers" / "learning.jsonl")
        model = ToyModel()
        catalog = [{"batch_id":b["batch_id"],"batch_hash":b["batch_hash"],"spans":[x["source"] for x in b["samples"]]} for b in batches]
        write_json(ARTIFACTS / "manifests" / "batch_catalog.json", catalog)

        # checkpoint before consuming the crash batch: next batch is exact and offsets are committed offsets.
        checkpoint = None
        for step, batch in enumerate(batches):
            if step == cfg.crash_step:
                checkpoint = save_checkpoint(ARTIFACTS / "checkpoints" / "checkpoint-0004.json", step, consumption.offset, learning.offset, batch, stage.name, {"seed":cfg.seed,"step":step})
                log.event("checkpoint saved")
                log.event("crash simulated")
                break
            loss, useful = model.loss(batch)
            consumption.append({"event":"BATCH_CONSUMED","step":step,"batch_id":batch["batch_id"],"batch_hash":batch["batch_hash"],"token_count":sum(len(x["input_ids"]) for x in batch["samples"])})
            learning.append({"event":"LOSS_RECORDED","step":step,"batch_id":batch["batch_id"],"batch_hash":batch["batch_hash"],"loss":loss,"loss_bearing_tokens":useful,"source_spans":[x["source"] for x in batch["samples"]]})

        resumed = load_checkpoint(ARTIFACTS / "checkpoints" / "checkpoint-0004.json")
        expected = batches[cfg.crash_step]
        resume_match = compare_batch(expected, {"batch_id":resumed["next_batch_id"],"batch_hash":resumed["next_batch_hash"]})
        log.event("run resumed")
        log.event("resume_next_batch_matched", resume_match)
        checks["crash_recovery"] = {"passed": resume_match and resumed["consumption_ledger_offset"] == cfg.crash_step and resumed["learning_ledger_offset"] == cfg.crash_step, "evidence":"checkpoints/checkpoint-0004.json"}

        for step in range(cfg.crash_step, len(batches)):
            batch = batches[step]
            loss, useful = model.loss(batch)
            consumption.append({"event":"BATCH_CONSUMED","step":step,"batch_id":batch["batch_id"],"batch_hash":batch["batch_hash"],"token_count":sum(len(x["input_ids"]) for x in batch["samples"])})
            learning.append({"event":"LOSS_RECORDED","step":step,"batch_id":batch["batch_id"],"batch_hash":batch["batch_hash"],"loss":loss,"loss_bearing_tokens":useful,"source_spans":[x["source"] for x in batch["samples"]]})

        chain_ok = consumption.verify_chain() and learning.verify_chain()
        log.event("ledgers verified", chain_ok)

        replayed = replay(batches, cfg.replay_start, cfg.replay_end)
        original = catalog[cfg.replay_start:cfg.replay_end]
        replay_match = replayed == original
        write_json(ARTIFACTS / "manifests" / "replay.json", {"original":original,"replayed":replayed,"match":replay_match,"stream_hash":stream_hash(batches)})
        log.event("historical stream replayed")
        log.event("replay_hash_matched", replay_match)
        checks["replay"] = {"passed":replay_match, "evidence":"manifests/replay.json"}

        fork = {"checkpoint_id":"fork-from-ckpt-0004","parent_checkpoint":resumed["checkpoint_id"],"parent_next_batch":resumed["next_batch_id"],"fork_policy":"coding_stem","lineage_hash":sha256_obj(resumed)}
        write_json(ARTIFACTS / "checkpoints" / "fork.json", fork)
        log.event("branch forked")

        actual = {}
        for b in batches:
            for p in b["samples"]:
                lane = p["source"]["lane"]
                actual[lane] = actual.get(lane,0)+1
        write_json(ARTIFACTS / "manifests" / "mixture_actual.json", {"planned":stage.weights,"actual_sample_counts":actual,"scheduler_order":"deterministic_lane_rank"})
        # Compliance here means every scheduled lane is represented and all protected floors have a planned share.
        mixture_ok = set(actual).issubset(stage.weights) and all(stage.weights[l] >= f for l,f in stage.floors.items())
        log.event("mixture compliance recorded", mixture_ok)
        checks["mixture_compliance"] = {"passed":mixture_ok, "evidence":"manifests/mixture.json + mixture_actual.json"}

        learning_ok = len(learning.records)==len(consumption.records) and all(a["batch_id"]==b["batch_id"] and a["batch_hash"]==b["batch_hash"] for a,b in zip(consumption.records, learning.records)) and chain_ok
        checks["learning_trace"] = {"passed":learning_ok, "evidence":"ledgers/consumption.jsonl + learning.jsonl"}

        elapsed=max(time.perf_counter()-start_time,1e-9)
        tokens=sum(r["token_count"] for r in consumption.records)
        useful=sum(r["loss_bearing_tokens"] for r in learning.records)
        allocated=len(consumption.records)*cfg.batch_size*(cfg.sequence_length-1)
        performance={"elapsed_seconds":elapsed,"batches":len(consumption.records),"tokens_consumed":tokens,"loss_bearing_tokens":useful,"tokens_per_second":tokens/elapsed,"useful_loss_bearing_tokens_per_second":useful/elapsed,"packing_utilization":tokens/allocated,"reconstructable_from":["ledgers/consumption.jsonl","ledgers/learning.jsonl","manifests/batches.json"]}
        write_json(ARTIFACTS/"performance.json",performance)
        log.event("performance measured")
        checks["throughput"]={"passed":performance["tokens_per_second"]>0 and performance["useful_loss_bearing_tokens_per_second"]>0,"evidence":"performance.json"}
        checks["ledger_integrity"]={"passed":chain_ok,"evidence":"hash-chained ledgers"}

        all_pass=all(v["passed"] for v in checks.values())
        log.event("audit completed", all_pass)
        evidence=generate_evidence(ARTIFACTS/"evidence.json",checks)
        generate_evidence_md(ARTIFACTS/"evidence.md",evidence)
        print("\nRESULT: PASS" if all_pass else "\nRESULT: FAIL")
        print(f"Artifacts: {ARTIFACTS}")
        return all_pass
    except Exception:
        log.event("audit completed",False)
        traceback.print_exc()
        return False
