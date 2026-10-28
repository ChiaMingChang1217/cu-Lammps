from pathlib import Path
from zipfile import ZipFile
import shutil
root=Path(__file__).resolve().parent
for archive in sorted((root/'archives').glob('production_*.zip')):
    with ZipFile(archive) as z:z.extractall(root)
for first in sorted((root/'archives/trajectories').glob('*.part01')):
    case=first.name.split('.lammpstrj.gz.part')[0]
    dst=root/'analysis/trajectories'/case/'trajectory.lammpstrj.gz'
    dst.parent.mkdir(parents=True,exist_ok=True)
    with dst.open('wb') as out:
        for part in sorted(first.parent.glob(case+'.lammpstrj.gz.part*')):
            with part.open('rb') as src:shutil.copyfileobj(src,out)
print('Restored 512 production folders and four compressed trajectories.')
