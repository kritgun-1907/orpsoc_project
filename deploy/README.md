# Running the full regeneration on EC2

Target instance for this project:

```
instance id    i-02415f8ffa4e0da4d   (orpsoc-compute-32)
type           c6i.8xlarge   32 vCPU / 64 GB
region         us-east-1
```

`c6i.8xlarge` is well matched to the workload. Seeds are the unit of
parallelism, step 7 runs **30 seeds** per level, and the pipeline picks
`ORPSOC_N_JOBS = 30` on a 32-vCPU box — every worker busy, two cores left for
the OS. A larger instance would idle, because there is no 31st seed to give it.

Memory is not the constraint: roughly 400 MB per worker, so ~12 GB of the 64 GB
is used. Disk should be **at least 20 GB** — the repo plus all datasets,
checkpoints, results and logs comes to well under 2 GB, but the OS image and
Python environment need room.

---

## Two things about this instance specifically

**It is currently Stopped.** Start it before running anything:

```bash
aws ec2 start-instances --region us-east-1 --instance-ids i-02415f8ffa4e0da4d
aws ec2 wait instance-running --region us-east-1 --instance-ids i-02415f8ffa4e0da4d
```

**It has no Elastic IP,** so its public DNS changes every stop/start cycle.
Do not hardcode a hostname — always drive the script by instance id and let it
resolve the current address:

```bash
./deploy/ec2_run.sh --instance-id i-02415f8ffa4e0da4d --key ~/.ssh/<your-key>.pem
```

If SSH times out, the security group is the usual cause: it needs inbound TCP 22
from your current IP, and your IP changes when you move networks.

---

## Usage

```bash
# full run: sync, execute, stream, fetch results
./deploy/ec2_run.sh --instance-id i-02415f8ffa4e0da4d --key ~/.ssh/k.pem

# Ctrl-C only detaches the log; the run continues on the instance
./deploy/ec2_run.sh --instance-id i-02415f8ffa4e0da4d --key ~/.ssh/k.pem --attach

# push code without running / pull results without running
./deploy/ec2_run.sh --instance-id i-02415f8ffa4e0da4d --key ~/.ssh/k.pem --sync-only
./deploy/ec2_run.sh --instance-id i-02415f8ffa4e0da4d --key ~/.ssh/k.pem --fetch-only
```

Add `--user ubuntu` if the AMI is Ubuntu rather than Amazon Linux.

---

## What the pipeline does, in order

1. install dependencies (skipped on re-runs via a marker file)
2. regenerate all synthetic datasets — v1 levels, L0 null, v2 benchmark
3. rebuild bonds + commodities, then sector-ETF + Fama-French, **from the
   shipped raw caches**
4. **integrity gate** — `verify_data.py`
5. **engine gate** — `test_equivalence.py` (18 assertions)
6. `step7_ablation.py` ← the expensive one
7. `step8_results.py`, `step9_real_data.py`
8. analyses: Jaccard-vs-null, adaptation test, compass ceiling
9. tar everything into `~/orpsoc_results_<timestamp>.tar.gz`

Any stage failing stops the pipeline; each writes its own log to `logs/`.

**Why the raw caches are shipped rather than re-downloaded.** `yfinance` returns
a different series on a different day — new bars, occasional restatements. If
the instance re-downloaded, its datasets would silently differ from the local
ones and nothing would be comparable. The four `raw_*.pkl` files travel with the
sync for that reason. Everything derived from them is regenerated on-instance.

**Why there is an integrity gate in front of the expensive stages.** This has
already bitten once: `data/fama_french.pkl` sat on disk for weeks holding labels
from a superseded target definition, agreeing with a target rebuilt from its own
`base` series only 47% of the time. Every experiment that read it measured the
wrong thing. `verify_data.py` reproduces each dataset's labels from its own
stored series and fails the run if they disagree.

---

## Resumability, and why spot is viable

`step7` and `step9` checkpoint per `(level, seed)` into
`results/checkpoints/<stage>/<provenance-hash>/`. Re-running the pipeline
resumes rather than restarting, so an interruption costs one seed.

That makes a **spot instance** reasonable here — roughly a third the price, and
the failure mode is one lost seed. The same-size spot capacity is requested
through a separate launch, not by changing this instance.

The provenance hash covers the run config **and** the source of
`orpsoc_utils.py` plus the runner. Change either and the old checkpoints are
correctly ignored instead of silently reloaded — this is guardrail G3 enforced
mechanically.

---

## Expected runtime and cost

Locally, 6 workers on a thermally-limited MacBook Air give ~3.8× throughput.
Thirty dedicated vCPUs should give ~28–30×, with no thermal ceiling.

```
step7 (5 levels x 30 seeds x 8 folds x 5 conditions)   ~45-60 min
step9 (2 datasets x 20 seeds)                          ~15-25 min
everything else                                          ~5-10 min
                                               total    ~1.5 h
```

At the us-east-1 on-demand rate for `c6i.8xlarge` (~$1.36/h) that is roughly
**$2 per full regeneration**.

---

## Results land in `ec2_results/`, not `results/`

The fetch writes to `ec2_results/` deliberately. An EC2 run must never silently
overwrite local numbers — diff first, then promote what you want:

```bash
python3 -c "
import json
a=json.load(open('results/step7_ablation.json'))
b=json.load(open('ec2_results/results/step7_ablation.json'))
print('local :', a['config'])
print('ec2   :', b['config'])
print('provenance match:', a.get('provenance',{}).get('hash')==b.get('provenance',{}).get('hash'))
"
```

---

## The instance keeps costing money after the run

`ec2_run.sh` never launches or terminates anything — provisioning and destroying
cloud resources is expensive to get wrong, so it only talks to a box you already
control. Stop it yourself when finished:

```bash
aws ec2 stop-instances --region us-east-1 --instance-ids i-02415f8ffa4e0da4d
```

`stop` preserves the root volume (you pay only EBS, a few cents a month) and is
what you want between runs. `terminate` destroys it.
