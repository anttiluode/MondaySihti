'use strict';
// Results-driven presentation: this page never simulates an upstream neuron.
(async function(){
const $=x=>document.getElementById(x);
const escape=x=>String(x).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let data;
try{const r=await fetch('./data/nonlinear_results.json');if(!r.ok)throw Error('HTTP '+r.status);data=await r.json();if(data.schema!=='mondaysihti.nonlinear.v1')throw Error('Unknown gate version');}
catch(e){$('g3-verdict').textContent='DATA NOT AVAILABLE';$('g3-explain').textContent=String(e);return;}
const methods=[['snapshot','Instantaneous snapshot'],['raw_three','Three raw soma readings'],['residue_three','Three Sihti residues'],['pca_three','Three PCA coordinates'],['full_eight','All eight readings (unmatched)']];
const pass=data.gates.G3_2_residue_5pct_better_than_matched;
$('g3-verdict').textContent=pass?'PASS':'FAIL';
let raw=data.methods.raw_three.mean_rmse,res=data.methods.residue_three.mean_rmse,pca=data.methods.pca_three.mean_rmse;
$('g3-explain').textContent=pass?'Stored residues beat both matched baselines at the preregistered margin.':`Residues: ${res.toFixed(3)} mV; raw: ${raw.toFixed(3)} mV; PCA: ${pca.toFixed(3)} mV. No 5% residue advantage on unseen history pairs.`;
for(const [id,key] of [['g3-g1','G3_1_exact_residue_inverse'],['g3-g3','G3_3_both_histories_5pct_better_than_snapshot']]){const ok=data.gates[key];$(id).textContent=ok?'PASS':'FAIL';$(id).classList.toggle('pass',ok);$(id).classList.toggle('fail',!ok);}
$('g3-upstream').textContent='The same original Operaattori AMPA/NMDA solver is exercised in upstream Actions, with a per-program numerical comparison.';
if(data.assumptions.independent_upstream_verification){$('g3-g0').textContent=data.gates.G3_0_original_source_verified?'PASS':'FAIL';$('g3-g0').classList.toggle('pass',data.gates.G3_0_original_source_verified);}
else {$('g3-g0').textContent='SEE CI';$('g3-upstream').textContent='The saved results use an independently cross-checked fast reference. Source fidelity and fresh rerun are documented in the linked GitHub Actions workflow; this file alone does not establish the upstream check.';}
const max=Math.max(...methods.map(([k])=>data.methods[k].mean_rmse))*1.06;
const root=$('g3-bars');root.replaceChildren();
for(const [key,title] of methods){const val=data.methods[key].mean_rmse;const row=document.createElement('div');row.className='bar-row '+(key==='residue_three'?'residue':key==='pca_three'?'pca':key==='full_eight'?'full':'');const name=document.createElement('div');name.className='label';name.textContent=title;const track=document.createElement('div');track.className='bar-track';const bar=document.createElement('div');bar.className='bar-fill';bar.style.width=(100*val/max).toFixed(1)+'%';track.append(bar);const metric=document.createElement('div');metric.className='bar-val';metric.textContent=val.toFixed(4);row.append(name,track,metric);root.append(row);}
const select=$('g3-seed');data.by_seed.forEach((row,i)=>{const o=document.createElement('option');o.value=i;o.textContent=String(row.seed);select.append(o);});
function chart(){const row=data.by_seed[Number(select.value)||0],ex=row.example;const times=data.probe_times_ms;const all=[...ex.response_actual,...ex.predictions.raw_three,...ex.predictions.residue_three,...ex.predictions.pca_three];const min=Math.min(...all),max=Math.max(...all),span=Math.max(.3,max-min);const y0=min-.15*span,y1=max+.15*span;const xp=i=>78+i*176,yp=v=>208-(v-y0)/(y1-y0)*167;const curves=[['actual',ex.response_actual,'#f8b886'],['raw',ex.predictions.raw_three,'#83b6fc'],['residue',ex.predictions.residue_three,'#79e1c0'],['pca',ex.predictions.pca_three,'#b49de6']];const svg=['<rect x="0" y="0" width="680" height="245" fill="#0e1a23"/>'];for(let k=0;k<=4;k++){const y=48+k*40;const value=y1-(y-41)/167*(y1-y0);svg.push(`<line x1="70" x2="640" y1="${y}" y2="${y}" stroke="#30404d" stroke-width="1"/><text x="10" y="${y+4}" fill="#b2c2cc" font-size="11">${value.toFixed(2)}</text>`);}curves.forEach(([name,vals,color])=>{svg.push(`<polyline points="${vals.map((v,i)=>xp(i)+','+yp(v)).join(' ')}" fill="none" stroke="${color}" stroke-width="${name==='actual'?3:2}" stroke-linecap="round" stroke-linejoin="round"/>`);for(let i=0;i<vals.length;i++)svg.push(`<circle cx="${xp(i)}" cy="${yp(vals[i])}" r="${name==='actual'?4:3}" fill="${color}"/>`);});times.forEach((t,i)=>svg.push(`<text x="${xp(i)}" y="234" text-anchor="middle" fill="#b0c1ca" font-size="11">${Number(t).toFixed(1)}ms</text>`));$('g3-trace').innerHTML=svg.join('');}
select.addEventListener('change',chart);chart();
})();
