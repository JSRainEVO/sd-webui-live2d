// Optional verification with the official Web Cubism Core supplied by the user.
// Downloaded Core stays in .tools and is not embedded in generated models.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

async function verify(manifestPath, corePath) {
  const context = {console, require, process, Buffer, setTimeout, clearTimeout,
    __dirname: path.dirname(path.resolve(corePath)), __filename: path.resolve(corePath),
    module: {exports: {}}, exports: {}};
  vm.createContext(context);
  vm.runInContext(fs.readFileSync(corePath, 'utf8'), context, {filename: 'cubism-core'});
  const core = context.Live2DCubismCore;
  let version;
  for (let i = 0; i < 200; i++) {
    try { version = core.Version.csmGetVersion(); break; }
    catch { await new Promise(resolve => setTimeout(resolve, 25)); }
  }
  if (!version) throw new Error('Cubism Core initialization timed out');
  const manifest = JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
  const bytes = fs.readFileSync(path.resolve(path.dirname(manifestPath), manifest.FileReferences.Moc));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const consistencyCheck = core.Moc.hasMocConsistency || core.Moc.prototype.hasMocConsistency;
  if (!consistencyCheck.call(core.Moc.prototype, buffer)) throw new Error('Official Core rejected MOC3 consistency');
  const moc = core.Moc.fromArrayBuffer(buffer);
  if (!moc) throw new Error('Official Core could not load MOC3');
  let model;
  try {
    model = core.Model.fromMoc(moc);
    if (!model) throw new Error('Official Core could not initialize model');
    function checkGeometry() {
      model.update();
      for (const vertices of model.drawables.vertexPositions)
        for (const value of vertices)
          if (!Number.isFinite(value)) throw new Error('Non-finite vertex in evaluated model');
    }
    checkGeometry();
    let poses = 1;
    for (let i = 0; i < model.parameters.count; i++) {
      for (const value of [model.parameters.minimumValues[i], model.parameters.maximumValues[i]]) {
        model.parameters.values.set(model.parameters.defaultValues);
        model.parameters.values[i] = value;
        checkGeometry();
        poses++;
      }
    }
    return {status: 'passed', coreVersion: version, consistency: true,
      initialized: true, parameters: model.parameters.count, drawables: model.drawables.count,
      finiteGeometryPoses: poses, renderedPlayback: 'not_tested'};
  } finally {
    if (model) model.release();
    moc._release();
  }
}

const [manifestPath, corePath, output] = process.argv.slice(2);
if (!manifestPath || !corePath) {
  console.error('Usage: node verify_model.cjs <model3.json> <official-Core.js> [report.json]');
  process.exitCode = 2;
} else {
  verify(manifestPath, corePath).then(result => {
    const json = JSON.stringify(result, null, 2);
    if (output) fs.writeFileSync(output, json);
    console.log(json);
  }).catch(error => {console.error(error.message); process.exitCode = 1;});
}
