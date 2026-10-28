from pathlib import Path
import os,sys,json
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parent/'ovito_packages'))
os.environ['QT_QPA_PLATFORM']='offscreen'
import numpy as np
import ovito
from ovito.io import import_file
from ovito.modifiers import CalculateDisplacementsModifier,PolyhedralTemplateMatchingModifier,ColorCodingModifier,ExpressionSelectionModifier,DeleteSelectedModifier
from ovito.pipeline import ReferenceConfigurationModifier
from ovito.vis import Viewport,TachyonRenderer
OUT=BASE/'figures'/'ovito';OUT.mkdir(parents=True,exist_ok=True)
records=[]
for folder in sorted((BASE/'trajectories').iterdir()):
 if not (folder/'rerun_report.json').exists():raise RuntimeError('Simulation incomplete: '+folder.name)
 report=json.loads((folder/'rerun_report.json').read_text());assert report['exit_code']==0
 pipeline=import_file(str(folder/'trajectory.lammpstrj.gz'))
 assert pipeline.source.num_frames==31
 ptm=PolyhedralTemplateMatchingModifier(rmsd_cutoff=0.1)
 pipeline.modifiers.append(ptm)
 disp=CalculateDisplacementsModifier(affine_mapping=ReferenceConfigurationModifier.AffineMapping.ToReference)
 pipeline.modifiers.append(disp)
 ref=pipeline.compute(0);L0=float(ref.cell[0,0])
 for frame in range(31):
  data=pipeline.compute(frame);types=np.asarray(data.particles['Structure Type']);u=np.asarray(data.particles['Displacement Magnitude'])
  records.append({'case_id':folder.name,'frame':frame,'strain':float(data.cell[0,0]/L0-1),'fcc_fraction':float((types==1).mean()),'hcp_fraction':float((types==2).mean()),'other_fraction':float((types==0).mean()),'displacement_mean_A':float(u.mean()),'displacement_p95_A':float(np.quantile(u,.95)),'natoms':len(u)})
 pipeline.add_to_scene()
 # Show a central 12-A slab through each current cell, not a surface projection.
 def slab(frame,data):
  z=np.asarray(data.particles.positions)[:,2];center=float(data.cell[2,3]+data.cell[2,2]/2)
  data.particles_.create_property('Selection',data=(np.abs(z-center)>6).astype(int))
 pipeline.modifiers.append(slab);pipeline.modifiers.append(DeleteSelectedModifier())
 pipeline.compute(30)
 vp=Viewport(type=Viewport.Type.Ortho,camera_dir=(0,0,-1))
 vp.zoom_all(size=(700,600));vp.fov*=1.1
 # Fixed camera and fixed color scale for all frames of this case.
 pipeline.source.data if False else None
 color=ColorCodingModifier(property='Displacement Magnitude',start_value=0,end_value=8,gradient=ColorCodingModifier.Viridis())
 pipeline.modifiers.append(color)
 for frame in [0,10,20,30]:
  vp.render_image(filename=str(OUT/f'{folder.name}_disp_{frame}.png'),size=(700,600),frame=frame,background=(1,1,1),renderer=TachyonRenderer(ambient_occlusion=True,shadows=False))
 pipeline.modifiers.remove(color)
 for frame in [0,10,20,30]:
  vp.render_image(filename=str(OUT/f'{folder.name}_ptm_{frame}.png'),size=(700,600),frame=frame,background=(1,1,1),renderer=TachyonRenderer(ambient_occlusion=True,shadows=False))
 pipeline.remove_from_scene()
 print('Rendered '+folder.name,flush=True)
(BASE/'ovito_metrics.json').write_text(json.dumps(records,indent=2))
print('OVITO '+ovito.version_string,flush=True)
