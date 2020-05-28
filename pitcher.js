// extracted from js library

var Pitcher = (function() {

  this.fileReader = new FileReader();

  this.sampleRate = 26040;
  this.semitones = 0;

  new Response(blob).arrayBuffer().then(function(result) {
    this.buffer = result;
    this.audioData = this.audioCtx.decodeAudioData(this.buffer)
    console.log(this.buffer)
    this.samples = this.buffer.getChannelData(0);
    this.baseSamples = [...this.samples];
    this.start = 0;
    this.end = this.samples.length;
  })



  var pitch = function(st) {
    let t;
    if (this._semitones = st, st < 0)
      switch (st) {
        case -1:
          t = 1.05652677103003;
          break;
        case -2:
          t = 1.1215356033380033;
          break;
        case -3:
          t = 1.1834835840896631;
          break;
        case -4:
          t = 1.253228360845465;
          break;
        case -5:
          t = 1.3310440397149297;
          break;
        case -6:
          t = 1.4039714929646099;
          break;
        case -7:
          t = 1.5028019735639886;
          break;
        case -8:
          t = 1.5766735700797954
      }
    else
      t = 1.02930223664 ** -st;
    const i = Math.round(this.baseSamples.length * t)
      , n = Math.round(this.end * t)
      , s = Math.round(this.start * t);
    this.buffer = this.audioCtx.createBuffer(1, i, this.sampleRate),
      this.samples = this.buffer.getChannelData(0);
    const r = linspace(0, this.baseSamples.length, i);
    for (let e = 0; e < n - s - 1; e++)
      this.samples[e] = this.baseSamples[Math.round(r[e + s])]
  }
})();
