'use strict';
const $ = (id) => document.getElementById(id);
const app = {domain:'image',view:'state',time:0,bench:'image',metric:'state',data:null};
const lin=(a,b,t)=>a+(b-a)*t;
const clamp=(v,a,b)=>Math.min(b,Math.max(a,v));
const rgb=(r,g,b)=>`rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
function ramp(v,max){const u=clamp(v/Math.max(1e-8,max),-1,1),a=Math.abs(u);return u>=0?rgb(lin(26,242,a),lin(48,171,a),lin(64,119,a)):rgb(lin(26,66,a),lin(48,120,a),lin(64,193,a));}
function renderField(){
 const d=app.data.illustration[app.domain];const canvas=$('field');const ctx=canvas.getContext('2d');const w=canvas.width,h=canvas.height;
 ctx.clearRect(0,0,w,h);ctx.fillStyle='#0a1219';ctx.fillRect(0,0,w,h);
 let vals=app.view==='residue'?d.bands[app.time]:d.states[app.time];
 if(app.view==='observed') vals=Array.from({length:d.states[0].length},()=>NaN);
 const isImage=app.domain==='image',n=isImage?8:7;
 if(app.view==='observed'){
   vals=d.observation[app.time];
   if(isImage){const idx=[0,7,18,27,36,45,56,63];const full=Array(64).fill(NaN);idx.forEach((k,i)=>full[k]=vals[i]);vals=full;}
   else{vals=[vals[0],NaN,NaN,NaN,NaN,NaN,NaN];}
 }
 const peak=Math.max(...vals.filter(Number.isFinite).map(Math.abs),0.001);
 if(isImage){const side=330,cell=side/8,ox=(w-side)/2,oy=(h-side)/2;for(let i=0;i<64;i++){const row=Math.floor(i/8),col=i%8;ctx.fillStyle=Number.isFinite(vals[i])?ramp(vals[i],peak):'#1c2c37';ctx.fillRect(ox+col*cell+1,oy+row*cell+1,cell-2,cell-2);if(!Number.isFinite(vals[i])){ctx.strokeStyle='#41505e';ctx.strokeRect(ox+col*cell+10,oy+row*cell+10,cell-20,cell-20)}}}
 else{ctx.lineWidth=9;ctx.strokeStyle='#395464';const nodes=[[130,190],[235,190],[295,115],[373,115],[445,105],[486,155],[280,265]];const edges=[[0,1],[1,2],[2,3],[3,4],[4,5],[1,6]];for(const [a,b] of edges){const A=nodes[a],B=nodes[b];ctx.beginPath();ctx.moveTo(A[0],A[1]);ctx.lineTo(B[0],B[1]);ctx.stroke()}nodes.forEach(([x,y],i)=>{ctx.beginPath();ctx.arc(x,y,i===0?27:23,0,2*Math.PI);ctx.fillStyle=Number.isFinite(vals[i])?ramp(vals[i],peak):'#293642';ctx.fill();ctx.strokeStyle=i===0?'#b4f0db':'#8297a3';ctx.lineWidth=i===0?3:1;ctx.stroke();ctx.fillStyle='#ecf3f3';ctx.font='13px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(i===0?'SOMA':String(i),x,y+5)});}
 $('field-title').textContent=(isImage?'IMAGE':'CABLE')+' · '+({state:'RELAXING STATE',residue:'OCTAVE RESIDUE',observed:'MEASURABLE OUTPUT'}[app.view]);
 $('field-caption').textContent=isImage?'Eight pixels are observable. The remaining 56 pixels are hidden from this observer.':'Only the soma (node 0) is observed. Other compartment voltages are hidden.';
 $('observer-desc').textContent=isImage?'Eight fixed pixel sensors; one row of the seven-checkpoint observation matrix.':'Only one soma sensor, read at seven checkpoints. No dendritic recordings.';
 $('time-value').textContent='t = '+app.data.times[app.time];
 const values=d.observation[app.time];const max=Math.max(...d.observation.flat().map(Math.abs),1e-6);const root=$('observations');root.innerHTML='';root.style.gridTemplateColumns=`repeat(${values.length},minmax(0,1fr))`;
 for(const v of values){const e=document.createElement('div');e.style.background=ramp(v,max);e.title=v.toFixed(4);root.append(e)}
 $('recon-error').textContent=d.reconstruction_error.toExponential(1);
}
function renderBench(){const run=app.data.cases[app.bench];const metric=app.metric==='state'?'mean_state_rmse':'mean_future_rmse';const entries=[['raw','Best 3 raw times'],['residue','Best 3 residues'],['pca','Train-fitted PCA'],['full_raw','All 7 raw'],['full_residue','All 7 residues']];const vals=entries.map(([key,label])=>({key,label,val:run.methods[key][metric]}));const maximum=Math.max(...vals.map(v=>v.val))*1.05;const root=$('bars');root.innerHTML='';for(const e of vals){const row=document.createElement('div');row.className='bar-row'+(e.key==='residue'?' residue':e.key==='pca'?' pca':e.key.startsWith('full')?' full':'');const label=document.createElement('div');label.className='label';label.textContent=e.label;const track=document.createElement('div');track.className='bar-track';const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=(100*e.val/maximum).toFixed(2)+'%';track.append(fill);const num=document.createElement('div');num.className='bar-val';num.textContent=e.val.toFixed(4);row.append(label,track,num);root.append(row)}$('metric-name').textContent=(app.metric==='state'?'HIDDEN STATE RMSE':'FUTURE SENSOR RMSE')+' · '+app.bench.toUpperCase();$('result-note').textContent=app.bench==='image'?'Image: 64 hidden pixel perturbations, 8 measured pixels, 7 times. State RMSE is in synthetic signal units.':'Cable: 7 hidden voltages, only 1 measured soma voltage, 7 times. RMSE is in mV.';
}
function setActive(selector,attr,value){document.querySelectorAll(selector).forEach(el=>{const yes=el.dataset[attr]===value;el.classList.toggle('active',yes);el.setAttribute('aria-pressed',String(yes))});}
function wire(){document.querySelectorAll('[data-domain]').forEach(el=>el.addEventListener('click',()=>{app.domain=el.dataset.domain;setActive('[data-domain]','domain',app.domain);renderField()}));document.querySelectorAll('[data-view]').forEach(el=>el.addEventListener('click',()=>{app.view=el.dataset.view;setActive('[data-view]','view',app.view);renderField()}));$('time-slider').addEventListener('input',e=>{app.time=Number(e.target.value);renderField()});document.querySelectorAll('[data-bench]').forEach(el=>el.addEventListener('click',()=>{app.bench=el.dataset.bench;setActive('[data-bench]','bench',app.bench);renderBench()}));document.querySelectorAll('[data-metric]').forEach(el=>el.addEventListener('click',()=>{app.metric=el.dataset.metric;setActive('[data-metric]','metric',app.metric);renderBench()}));}
async function init(){try{const response=await fetch('./data/results.json');if(!response.ok)throw Error('HTTP '+response.status);app.data=await response.json();if(app.data.schema!=='mondaysihti.bridge.v1')throw Error('Unknown data version');const gate=app.data.gates.G2_residue_beats_both_matched_baselines_both_domains;$('result-verdict').textContent=gate?'PASSED':'FAILED';$('result-text').textContent=gate?'Residue compression won both matched cross-domain tests.':'Residues did not outperform equally sized raw/PCA encodings in both systems. The new axis adds no guaranteed recovery.';for(const [id,key] of [['g0','G0_exact_telescope'],['g1','G1_observability_rank_preserved'],['g2','G2_residue_beats_both_matched_baselines_both_domains']]){const elem=$(id),ok=app.data.gates[key];elem.textContent=ok?'PASS':'FAIL';elem.classList.toggle('pass',ok);elem.classList.toggle('fail',!ok)}wire();renderField();renderBench();}catch(e){$('result-verdict').textContent='DATA ERROR';$('result-text').textContent='Failed to load the reproducible results: '+e.message;console.error(e)}}
init();
