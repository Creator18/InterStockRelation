"""GPU lease: start a run on whatever GPU is free, hand off to a better GPU when one frees up.

One "owner" job trains a run at a time. A job that starts on a better GPU writes a takeover
request; the owner checkpoints at its next check and releases; the new job resumes from the
checkpoint. On a Slurm time-limit warning (SIGUSR1/SIGTERM) the owner checkpoints and requeues.

    lease = Lease(run, out_dir, dev)      # exits if the run is done or a better GPU owns it
    ...  lease.beat(); if lease.should_yield(): lease.release(save_fn)
    lease.finish(cancel_prefix='isr_t11_')
"""
import os, json, time, signal
import torch

TIERS = [('H200', 9), ('H100', 8), ('A100', 7), ('L40', 6), ('A6000', 5), ('A5000', 4), ('V100', 3), ('T4', 2), ('P100', 1)]
STALE = 900


class Lease:
    def __init__(self, run, out_dir, dev):
        self.run, self.dev = run, dev
        self.job = os.environ.get('SLURM_JOB_ID', f'local{os.getpid()}')
        self.path = os.path.join(out_dir, f'{run}.lease.json')
        self.tier = self._tier()
        self.stop = False
        for s in ('SIGUSR1', 'SIGTERM'):
            if hasattr(signal, s): signal.signal(getattr(signal, s), lambda *_: setattr(self, 'stop', True))
        cur = self._rd()
        if cur.get('state') == 'done': print('run already finished', flush=True); raise SystemExit(0)
        if cur.get('state') == 'running' and time.time() - cur.get('beat', 0) < STALE and cur.get('job') != self.job:
            if self.tier <= cur.get('tier', 0):
                print(f"owner {cur['job']} has an equal/better GPU; exiting", flush=True); raise SystemExit(0)
            cur['request'] = dict(job=self.job, tier=self.tier); self._wr(cur)
            print(f"requesting takeover from {cur['job']}", flush=True)
            t0 = time.time()
            while time.time() - t0 < 1800:
                c = self._rd()
                if c.get('state') != 'running' or time.time() - c.get('beat', 0) > STALE: break
                time.sleep(15)
        self.beat(); print(f'lease acquired job={self.job} tier={self.tier}', flush=True)

    def _tier(self):
        if self.dev.type != 'cuda': return 0
        pr = torch.cuda.get_device_properties(0); nm = pr.name.upper()
        return next((r for k, r in TIERS if k in nm), 1) * 1000 + pr.total_memory // 2**30

    def _rd(self):
        try: return json.load(open(self.path))
        except Exception: return {}

    def _wr(self, d):
        tmp = self.path + f'.{self.job}'; json.dump(d, open(tmp, 'w')); os.replace(tmp, self.path)

    def beat(self, state='running'):
        req = self._rd().get('request')                 # keep a pending upgrade request intact
        d = dict(job=self.job, tier=self.tier, state=state, beat=time.time(),
                 gpu=torch.cuda.get_device_name(0) if self.dev.type == 'cuda' else 'cpu')
        if req and req.get('job') != self.job and state == 'running': d['request'] = req
        self._wr(d)

    def should_yield(self):
        if self.stop: return 'signal'
        r = self._rd().get('request')
        return 'upgrade' if r and r.get('tier', 0) > self.tier else None

    def release(self, save_fn, reason):
        save_fn(); c = self._rd(); c.update(state='released', beat=time.time()); c.pop('request', None); self._wr(c)
        print(f'checkpointed and released lease ({reason})', flush=True)
        if reason == 'signal' and 'SLURM_JOB_ID' in os.environ: os.system(f'scontrol requeue {self.job}')
        raise SystemExit(0)

    def finish(self, cancel_prefix):
        self.beat('done')
        if 'SLURM_JOB_ID' in os.environ: os.system(f'scancel --me --name={cancel_prefix}{self.run} --state=PENDING')
