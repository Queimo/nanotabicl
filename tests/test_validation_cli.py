import subprocess, sys, json, pathlib

def test_validation_cli(tmp_path):
    out=tmp_path/'v.json'; r=subprocess.run([sys.executable,'validate_prior.py','--seed','17','--episodes','2','--output',str(out)],capture_output=True,text=True)
    assert r.returncode==0, r.stderr+r.stdout
    assert json.loads(out.read_text())['summary']['passed']
