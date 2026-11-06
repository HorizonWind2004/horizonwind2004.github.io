// Choreography ported from OmniTaskonomy's components/tldr-teaser.tsx.
// https://omni-taskonomy.github.io/ (September 30, 2026)
const labels = { i2t_answer: "('reorder', [3, 2, 1, 0])" };
const permutation = [3, 2, 1, 0];
const answer = Array.from(labels.i2t_answer);
const RING = 2 * Math.PI * 15;
const root = document.querySelector('.tldr-teaser');
function tile(index, position, extra = '') {
  const element = document.createElement('span');
  element.className = 'jigsaw-tile ' + extra;
  element.dataset.pos = position;
  element.style.gridArea = `${(position >> 1) + 1} / ${(position & 1) + 1}`;
  element.style.backgroundImage = `url('jigsaw-tile-${index}.webp')`;
  const ring = document.createElement('span');
  ring.className = 'jigsaw-ring';
  const badge = document.createElement('span');
  badge.className = 'jigsaw-badge';
  badge.textContent = index;
  element.append(ring, badge);
  return element;
}
root.querySelectorAll('.jigsaw-input').forEach(grid => {
  for (let index = 0; index < 4; index++) grid.append(tile(index, index));
});
const output = root.querySelector('.jigsaw-output');
permutation.forEach((index, position) => {
  const slot = document.createElement('span');
  slot.className = 'jigsaw-slot';
  slot.style.gridArea = `${(position >> 1) + 1} / ${(position & 1) + 1}`;
  output.append(slot, tile(index, position, 'jigsaw-flyer'));
});
const code = root.querySelector('.teaser-answer code');
answer.forEach(char => {
  const span = document.createElement('span');
  span.className = 'answer-char';
  span.textContent = char;
  code.append(span);
});
function schedule() {
    const reset = [650, 1100], cue = 1150, arrow = [1450, 1900];
    const fly = 1950, stagger = 250, flight = 800;
    const landed = fly + 3 * stagger + flight;
    const settle = [landed + 40, landed + 440];
    const shine = [settle[0] + 160, settle[0] + 980];
    const transfer = [settle[1] + 260, settle[1] + 1260];
    const understand = transfer[1] - 280;
    const arrow2 = [understand + 320, understand + 770];
    const box = [arrow2[1] - 40, arrow2[1] + 260];
    const reveal = [];
    const highlight = {};
    let t = box[1] + 180;
    for (const char of answer) {
        if (/\d/.test(char)) {
            const start = t;
            t += 300;
            reveal.push(t);
            highlight[Number(char)] = [start, t + 460];
            t += 110;
        }
        else {
            reveal.push(t);
            t += char === ' ' ? 12 : 58;
        }
    }
    const done = t, together = [done + 260, done + 720];
    return { reset, cue, arrow, fly, stagger, flight, settle, shine, transfer, understand, arrow2, box, reveal, highlight, done, together, length: done + 2500 };
}
const EASE = 'cubic-bezier(.45,0,.2,1)', OUT = 'cubic-bezier(.2,.8,.2,1)', FLY = 'cubic-bezier(.6,0,.22,1)';
const RISE = 'cubic-bezier(.25,.6,.45,1)', FALL = 'cubic-bezier(.55,0,.75,.4)';
function build(root) {
    const plan = schedule(), L = plan.length, animations = [];
    const all = (selector) => [...root.querySelectorAll(selector)];
    const one = (selector) => root.querySelector(selector);
    const run = (element, frames) => {
        if (!element)
            return;
        const keyframes = frames.map(([at, style, easing]) => ({ ...style, offset: Math.min(1, at / L), ...(easing ? { easing } : {}) }));
        if (keyframes[0].offset !== 0)
            keyframes.unshift({ ...frames[0][1], offset: 0 });
        if (keyframes[keyframes.length - 1].offset !== 1)
            keyframes.push({ ...frames[frames.length - 1][1], offset: 1 });
        animations.push(element.animate(keyframes, { duration: L, iterations: Infinity, fill: 'both' }));
    };
    // Hold a, ease to b over [from, to]; optionally return to a over [back, backTo].
    const swing = (a, b, from, to, back, backTo, easing = EASE) => [[0, a], [from, a, easing], [to, b], ...(back !== undefined ? [[back, b, easing], [backTo, a]] : [])];
    const box = (element) => element.getBoundingClientRect();
    const [r0, r1] = plan.reset;
    const i2i = one('.teaser-i2i'), i2t = one('.teaser-i2t');
    run(i2i, swing({ opacity: 1 }, { opacity: .58 }, plan.transfer[0] + 320, plan.transfer[0] + 720, plan.together[0], plan.together[1]));
    run(i2i?.querySelector('.teaser-glow') ?? null, swing({ opacity: 0 }, { opacity: 1 }, plan.cue - 120, plan.cue + 260, plan.transfer[0] + 320, plan.transfer[0] + 720));
    run(i2t, swing({ opacity: 1 }, { opacity: .5 }, r0, r1, plan.understand, plan.understand + 420));
    run(i2t?.querySelector('.teaser-glow') ?? null, [[0, { opacity: 0 }], [plan.understand, { opacity: 0 }, EASE], [plan.understand + 420, { opacity: 1 }], [plan.together[0], { opacity: 1 }, EASE], [plan.together[1], { opacity: 0 }]]);
    // Arrows retract during the reset and draw again when each task starts.
    const drawArrow = (panel, [a0, a1]) => {
        run(panel?.querySelector('.teaser-arrow i') ?? null, [[0, { transform: 'scaleX(1)' }], [r0, { transform: 'scaleX(1)' }, EASE], [r1, { transform: 'scaleX(0)' }], [a0, { transform: 'scaleX(0)' }, OUT], [a1, { transform: 'scaleX(1)' }]]);
        run(panel?.querySelector('.teaser-arrow b') ?? null, [[0, { opacity: 1, transform: 'translateX(0) rotate(45deg)' }], [r0, { opacity: 1, transform: 'translateX(0) rotate(45deg)' }, EASE], [r1, { opacity: 0, transform: 'translateX(-6px) rotate(45deg)' }], [a1 - 160, { opacity: 0, transform: 'translateX(-6px) rotate(45deg)' }, OUT], [a1 + 60, { opacity: 1, transform: 'translateX(0) rotate(45deg)' }]]);
    };
    drawArrow(i2i, plan.arrow);
    drawArrow(i2t, plan.arrow2);
    // Input patches breathe once as each task begins; I2T patches also lift while their index is typed.
    const pulse = (start, peak, end) => [[start, { transform: 'scale(1)' }, EASE], [peak, { transform: 'scale(1.06)' }], [end - 180, { transform: 'scale(1.06)' }, EASE], [end, { transform: 'scale(1)' }]];
    // A scaled patch is raised above its neighbours for the whole pulse; later pulses sit higher when they overlap.
    const raise = (windows) => [[0, { zIndex: 'auto' }], ...windows.flatMap(([start, end, z]) => [[start, { zIndex: 'auto' }], [start, { zIndex: z }], [end, { zIndex: z }], [end, { zIndex: 'auto' }]])];
    const highlightOrder = Object.values(plan.highlight).map(([start]) => start).sort((a, b) => a - b);
    all('.teaser-i2i .jigsaw-input .jigsaw-tile').forEach((tile, k) => {
        const breathe = plan.cue + 70 * k;
        run(tile, [[0, { transform: 'scale(1)' }], ...pulse(breathe, breathe + 220, breathe + 520)]);
        run(tile, raise([[breathe, breathe + 520, 2 + k]]));
    });
    all('.teaser-i2t .jigsaw-input .jigsaw-tile').forEach((tile, k) => {
        const [h0, h1] = plan.highlight[k];
        const breathe = plan.understand + 120 + 70 * k;
        run(tile, [[0, { transform: 'scale(1)' }], ...pulse(breathe, breathe + 220, breathe + 520), ...pulse(h0, h0 + 200, h1)]);
        run(tile, raise([[breathe, breathe + 520, 2 + k], [h0, h1, 6 + highlightOrder.indexOf(h0)]]));
        run(tile.querySelector('.jigsaw-ring'), [[0, { opacity: 0 }], [h0, { opacity: 0 }, EASE], [h0 + 180, { opacity: 1 }], [h1 - 160, { opacity: 1 }, EASE], [h1, { opacity: 0 }]]);
        run(tile.querySelector('.jigsaw-badge'), [[0, { backgroundColor: 'rgba(22,34,46,.74)' }], [h0, { backgroundColor: 'rgba(22,34,46,.74)' }, EASE], [h0 + 180, { backgroundColor: 'rgba(56,118,176,.96)' }], [h1 - 160, { backgroundColor: 'rgba(56,118,176,.96)' }, EASE], [h1, { backgroundColor: 'rgba(22,34,46,.74)' }]]);
    });
    // I2I: each patch leaves its shuffled cell and arcs into its ordered cell, then the supplied output resolves.
    const inputs = all('.teaser-i2i .jigsaw-input .jigsaw-tile'), slots = all('.teaser-i2i .jigsaw-slot');
    all('.teaser-i2i .jigsaw-flyer').forEach((flyer, p) => {
        const from = box(inputs[permutation[p]]), to = box(slots[p]);
        const dx = from.left - to.left, dy = from.top - to.top;
        const s = plan.fly + p * plan.stagger, e = s + plan.flight;
        // The path eases as one move; the lift rises and falls independently so the arc never stalls.
        run(flyer, [[0, { translate: `${dx}px ${dy}px` }], [s, { translate: `${dx}px ${dy}px` }, FLY], [e, { translate: '0px 0px' }]]);
        const rest = { transform: 'translateY(0px) scale(1)', boxShadow: '0 0 0 rgba(20,36,52,0)' };
        run(flyer, [[0, rest], [s, rest, RISE], [s + plan.flight * .48, { transform: 'translateY(-18px) scale(1.12)', boxShadow: '0 16px 26px rgba(20,36,52,.26)' }, FALL], [e, { transform: 'translateY(0px) scale(1)', boxShadow: '0 1px 2px rgba(20,36,52,.12)' }]]);
        run(flyer, [[0, { opacity: 0 }], [s, { opacity: 0 }], [s + 50, { opacity: 1 }], [plan.settle[1] + 60, { opacity: 1 }], [plan.settle[1] + 120, { opacity: 0 }]]);
        run(flyer.querySelector('.jigsaw-badge'), [[0, { opacity: 1 }], [s + plan.flight * .3, { opacity: 1 }, EASE], [e - 100, { opacity: 0 }]]);
    });
    slots.forEach(slot => run(slot, [[0, { opacity: 0 }], [r0, { opacity: 0 }, EASE], [r1, { opacity: 1 }], [plan.settle[0], { opacity: 1 }, EASE], [plan.settle[1], { opacity: 0 }]]));
    run(one('.jigsaw-result'), [[0, { opacity: 1, transform: 'scale(1)' }], [r0, { opacity: 1, transform: 'scale(1)' }, EASE], [r1, { opacity: 0, transform: 'scale(.96)' }], [plan.settle[0], { opacity: 0, transform: 'scale(1.02)' }, OUT], [plan.settle[1], { opacity: 1, transform: 'scale(1)' }]]);
    run(one('.jigsaw-shine'), [[0, { opacity: 0, transform: 'translateX(-70%)' }], [plan.shine[0], { opacity: 0, transform: 'translateX(-70%)' }, EASE], [plan.shine[0] + 160, { opacity: 1, transform: 'translateX(-40%)' }], [plan.shine[1] - 160, { opacity: 1, transform: 'translateX(40%)' }, EASE], [plan.shine[1], { opacity: 0, transform: 'translateX(70%)' }]]);
    // Ability transfer: a red-to-blue trace runs from generation to understanding.
    const line = one('.ability-arrow > i');
    if (line) {
        const size = box(line), vertical = size.height > size.width;
        const [t0, t1] = plan.transfer;
        const axis = vertical ? 'scaleY' : 'scaleX';
        const travel = vertical ? `translate(0px, ${size.height}px)` : `translate(${size.width}px, 0px)`;
        run(line.querySelector('.ability-flow'), [[0, { opacity: 1, transform: `${axis}(1)` }], [r0, { opacity: 1, transform: `${axis}(1)` }, EASE], [r1, { opacity: 0, transform: `${axis}(1)` }], [r1 + 10, { opacity: 0, transform: `${axis}(0)` }], [t0, { opacity: 1, transform: `${axis}(0)` }, EASE], [t1, { opacity: 1, transform: `${axis}(1)` }]]);
        run(line.querySelector('.ability-comet'), [[0, { transform: 'translate(0px, 0px)' }], [t0, { transform: 'translate(0px, 0px)' }, EASE], [t1, { transform: travel }]]);
        run(line.querySelector('.ability-comet'), [[0, { opacity: 0 }], [t0, { opacity: 0 }], [t0 + 120, { opacity: 1 }], [t1 - 60, { opacity: 1 }, EASE], [t1 + 220, { opacity: 0 }]]);
    }
    run(one('.ability-generation'), swing({ opacity: 1 }, { opacity: .34 }, plan.transfer[0] + 520, plan.transfer[0] + 920, plan.together[0], plan.together[1]));
    run(one('.ability-understanding'), swing({ opacity: 1 }, { opacity: .34 }, r0, r1, plan.transfer[1] - 360, plan.transfer[1] + 60));
    run(one('.ability-arrow > span'), swing({ opacity: 1 }, { opacity: .5 }, r0, r1, plan.transfer[0] - 160, plan.transfer[0] + 240));
    // I2T: the supplied answer is typed into the output box, one existing character at a time.
    run(one('.teaser-answer'), [[0, { opacity: 1, transform: 'translateY(0px) scale(1)' }], [r0, { opacity: 1, transform: 'translateY(0px) scale(1)' }, EASE], [r1, { opacity: 0, transform: 'translateY(4px) scale(.97)' }], [plan.box[0], { opacity: 0, transform: 'translateY(4px) scale(.97)' }, OUT], [plan.box[1], { opacity: 1, transform: 'translateY(0px) scale(1)' }]]);
    const chars = all('.answer-char'), answerBox = one('.teaser-answer'), caret = one('.answer-caret');
    chars.forEach((char, i) => {
        const at = plan.reveal[i];
        run(char, [[0, { opacity: 1 }], [r1, { opacity: 1 }], [r1, { opacity: 0 }], [at, { opacity: 0 }], [at + 70, { opacity: 1 }]]);
        if (/\d/.test(answer[i]))
            run(char, [[0, { color: '#263b4d' }], [at, { color: '#263b4d' }], [at, { color: '#2f73b3' }], [at + 520, { color: '#2f73b3' }, EASE], [at + 1100, { color: '#263b4d' }]]);
    });
    if (answerBox && caret && chars.length) {
        const origin = box(answerBox), border = parseFloat(getComputedStyle(answerBox).borderLeftWidth) || 0;
        const place = (x, y) => ({ transform: `translate(${x - origin.left - border}px, ${y - origin.top - border}px)` });
        const first = chars[0].getClientRects()[0] ?? box(chars[0]);
        const ends = chars.map(char => { const rects = char.getClientRects(), rect = rects[rects.length - 1] ?? box(char); return place(rect.right, rect.top); });
        caret.style.height = `${first.height}px`;
        const moves = [[0, ends[ends.length - 1]], [r0, ends[ends.length - 1]], [r1, place(first.left, first.top)]];
        chars.forEach((_, i) => { moves.push([plan.reveal[i], moves[moves.length - 1][1]], [plan.reveal[i], ends[i]]); });
        run(caret, moves);
        const blink = [];
        const toggle = (from, to) => { for (let t = from, on = true; t < to; t += 530, on = !on)
            blink.push([t, { opacity: on ? 1 : 0 }], [Math.min(t + 530, to), { opacity: on ? 1 : 0 }]); };
        toggle(0, r0);
        blink.push([r0, { opacity: 0 }], [plan.box[1], { opacity: 0 }], [plan.box[1], { opacity: 1 }]);
        toggle(plan.done, L);
        run(caret, blink);
    }
    run(one('.teaser-progress'), [[0, { strokeDashoffset: `${RING}` }], [L, { strokeDashoffset: '0' }]]);
    return animations;
}

const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
let animations = [];
let ready = false;
function sync() {
  animations.forEach(animation => document.hidden ? animation.pause() : animation.play());
}
function start() {
  animations.forEach(animation => animation.cancel());
  animations = [];
  if (ready && !reducedMotion.matches && typeof Element.prototype.animate === 'function') {
    animations = build(root);
    sync();
  }
}
Promise.all([...permutation.map(index => `jigsaw-tile-${index}.webp`), 'jigsaw-output.webp'].map(src => {
  const image = new Image();
  image.src = src;
  return image.decode().catch(() => {});
})).then(() => {
  ready = true;
  start();
});
reducedMotion.addEventListener('change', start);
document.addEventListener('visibilitychange', sync);
