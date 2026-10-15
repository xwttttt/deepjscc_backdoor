"""Fetch a small pinned official-source snapshot, without executing upstream code."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT='b342d7ffdb552aba49b689cc50e0b1eacd29d042'
SELECTED=['README.md','requirements.txt','black_box_attack.py','evaluate_SDR.py',
          'configs/config.py','configs/config_attack.py','configs/config_ucf.py',
          'models/generator.py','models/channel.py','models/__init__.py',
          'models/JSCCOFDM_model.py','models/attack_basics.py','models/networks.py']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    tree=json.load(urllib.request.urlopen(
        f'https://api.github.com/repos/juc023/Magmaw/git/trees/{COMMIT}?recursive=1',timeout=30))
    assert tree['sha']==COMMIT
    indexed={item['path']:item for item in tree['tree'] if item['type']=='blob'}
    manifest={'repository':'https://github.com/juc023/Magmaw','commit':COMMIT,
              'kind':'read_only_core_source_snapshot_not_runnable_checkout','files':{},'errors':{}}
    for name in SELECTED:
        item=indexed[name]
        try:
            value=json.load(urllib.request.urlopen(
                'https://api.github.com/repos/juc023/Magmaw/git/blobs/'+item['sha'],timeout=30))
            data=base64.b64decode(value['content'])
            assert hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()==item['sha']
            target=root/name
            assert target.resolve().is_relative_to(root)
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(data)
            manifest['files'][name]={'git_blob_sha':item['sha'],'sha256':hashlib.sha256(data).hexdigest()}
            print('verified',name,flush=True)
        except Exception as error:
            manifest['errors'][name]=str(error)
            print('failed',name,type(error).__name__,flush=True)
            if len(manifest['errors'])>=2:
                break
    (root/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if manifest['errors']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
