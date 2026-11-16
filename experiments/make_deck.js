// Presentation in the style of the team's 딥러닝.pdf (white slide, coral frame, section tag, bracketed title, coral table headers).
// Numbers come from deck_data.json (make_deck_data.py).  Run:  node make_deck.js   -> dynamic_seRNN_slides.pptx
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const D = JSON.parse(fs.readFileSync("deck_data.json", "utf-8"));

const CORAL = "FF5A44", PINK = "FFE1DC", GREY = "F0F0F0", ZEBRA = "F5F5F5", INK = "1F1F1F", MUTED = "6B7280";
const FONT = "Malgun Gothic";
const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 (16:9, same ratio as the team's PDF)
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };
pres.title = "동적 좌표 seRNN - 중간 결과와 병목 진단";

const f3 = (x) => (x === null || x === undefined ? "-" : x.toFixed(3));
const f2 = (x) => (x === null || x === undefined ? "-" : x.toFixed(2));
const f0 = (x) => (x === null || x === undefined ? "-" : Math.round(x).toString());
const pm = (t) => (!t || t[0] === null ? "-" : t[2] > 1 ? `${t[0].toFixed(3)} ± ${t[1].toFixed(3)}` : t[0].toFixed(3));
const nOf = (t) => (t && t[2] ? t[2] : 0);

function frame(slide, sec, secName, title, notes) {
  slide.background = { color: "FFFFFF" };
  slide.addShape(pres.shapes.RECTANGLE, { x: 0.07, y: 0.07, w: 13.19, h: 7.36, fill: { type: "none" }, line: { color: CORAL, width: 10 }, objectName: "frame" });
  if (sec) {
    slide.addText([{ text: sec + "  ", options: { color: CORAL, bold: true } }, { text: secName, options: { color: INK, bold: true } }],
      { x: 0.55, y: 0.38, w: 6, h: 0.4, fontSize: 18, fontFace: FONT, margin: 0, isTextBox: true, objectName: "section tag" });
  }
  // bracketed title (as in the team's slides)
  const bx1 = 2.2, bx2 = 11.13, by1 = 0.62, by2 = 1.42;
  for (const [x, dir] of [[bx1, 1], [bx2, -1]]) {
    slide.addShape(pres.shapes.LINE, { x, y: by1, w: 0, h: by2 - by1, line: { color: CORAL, width: 1.5 } });
    slide.addShape(pres.shapes.LINE, { x: dir === 1 ? x : x - 0.14, y: by1, w: 0.14, h: 0, line: { color: CORAL, width: 1.5 } });
    slide.addShape(pres.shapes.LINE, { x: dir === 1 ? x : x - 0.14, y: by2, w: 0.14, h: 0, line: { color: CORAL, width: 1.5 } });
  }
  slide.addText(title, { x: 2.4, y: 0.62, w: 8.53, h: 0.8, fontSize: 32, bold: true, align: "center", valign: "middle", color: INK, fontFace: FONT, margin: 0, isTextBox: true, objectName: "title" });
  if (notes) slide.addNotes(notes);
}

function table(slide, rows, opt) {
  // rows[0] = header strings; others = arrays of strings or {t, hi:true}
  const body = rows.slice(1).map((r, i) =>
    r.map((c, j) => {
      const cell = typeof c === "object" ? c : { t: c };
      return { text: String(cell.t), options: { fill: { color: i % 2 === 0 ? "FFFFFF" : ZEBRA }, color: cell.hi ? CORAL : INK, bold: !!cell.hi || j === 0, align: j === 0 ? "center" : "center", valign: "middle" } };
    }));
  const head = rows[0].map((c) => ({ text: c, options: { fill: { color: CORAL }, color: "FFFFFF", bold: true, align: "center", valign: "middle" } }));
  slide.addTable([head, ...body], Object.assign({ fontFace: FONT, fontSize: 13, border: { type: "solid", pt: 0.5, color: "E5E5E5" }, rowH: 0.42, autoPage: false }, opt));
}

function box(slide, x, y, w, h, heading, items, opt = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: GREY }, line: { color: GREY, width: 0 }, rectRadius: 0.12, objectName: "box" });
  const runs = [{ text: heading, options: { fontSize: opt.hsize || 20, bold: true, color: opt.hcolor || INK, breakLine: true, paraSpaceAfter: 6 } }];
  items.forEach((t, i) => runs.push({ text: t, options: { bullet: true, fontSize: opt.size || 14, color: INK, breakLine: i < items.length - 1, paraSpaceAfter: 6 } }));
  slide.addText(runs, { x: x + 0.25, y: y + 0.15, w: w - 0.5, h: h - 0.3, fontFace: FONT, valign: "top", margin: 0, isTextBox: true });
}

function foot(slide, text) {
  slide.addText(text, { x: 0.55, y: 6.92, w: 12.2, h: 0.35, fontSize: 11, color: MUTED, fontFace: FONT, margin: 0, isTextBox: true, objectName: "footnote" });
}

// ------------------------------------------------------------------------------------------------------------------ 1 title
{
  const s = pres.addSlide();
  frame(s, "", "", "동적 좌표 seRNN", "뉴런이 3D 격자에서 시작해 움직일 때의 결과와, 현재 파이프라인에서 정확도가 0.4에 머무는 이유를 정리했습니다.");
  s.addText("3D 격자에서 시작해 뉴런이 움직일 때의 결과와 현재 병목", { x: 1.2, y: 2.4, w: 10.9, h: 0.7, fontSize: 24, align: "center", color: INK, fontFace: FONT, isTextBox: true });
  s.addText("CLEVRER 3000개 영상 - 학습 0-1999 / 검증 2000-2499 / 평가 2500-2999 - 서술형", { x: 1.2, y: 3.3, w: 10.9, h: 0.5, fontSize: 16, align: "center", color: MUTED, fontFace: FONT, isTextBox: true });
  box(s, 2.2, 4.4, 8.93, 1.9, "발표 내용", ["동적 이동 방식과 이동 후 뉴런 배치", "모듈, 스몰월드 구조는 어디서 생기는가", "정확도가 0.4에 머무는 원인과 해결 방향"], { hcolor: CORAL, size: 16, hsize: 20 });
}

// ------------------------------------------------------------------------------------------------------------------ 2 method
{
  const s = pres.addSlide();
  frame(s, "02", "연구 방법", "동적 3D 좌표", "정적 실험과 같은 5x5x4 격자에서 시작합니다. position은 좌표를 학습하고, swap은 뉴런 자리를 교환하며, relayout은 한 번만 재배치합니다.");
  box(s, 0.6, 1.8, 5.0, 4.9, "핵심 아이디어", [
    "5x5x4 격자(뉴런 100개)에서 시작 - 정적 실험과 같은 출발점",
    "손실 = 과제 손실 + 거리·통신 벌점 + 최소거리 반발",
    "연결이 세고 먼 뉴런 쌍은 서로 가까워지도록 이동",
    "최소거리 0.5 미만이면 벌점을 줘서 한 점에 뭉치는 것을 방지 (엄격한 제약은 아님)",
    "이동은 첫 에포크부터 열음 (open_epoch 1)"], { size: 14, hsize: 22, hcolor: CORAL });
  s.addText("동적 이동 방식", { x: 6.0, y: 1.8, w: 6.7, h: 0.4, fontSize: 18, bold: true, fontFace: FONT, margin: 0, isTextBox: true });
  table(s, [["방식", "무엇이 움직이나", "설정"],
    ["고정", "이동 없음 (정적과 같음)", "기준선"],
    ["position", "좌표를 Adam으로 학습", "학습률 0.01, 반발 0.01"],
    ["swap", "에포크마다 뉴런 쌍 교환", "배선 비용이 줄 때만, 10회"],
    ["both", "position + swap", "둘 다"],
    ["relayout", "10에포크 후 한 번 재배치", "이후 고정"]], { x: 6.0, y: 2.3, w: 6.7, colW: [1.4, 2.8, 2.5], rowH: 0.62 });
  foot(s, "평가: 같은 데이터·분할로 고정, position, swap, both, relayout을 비교하고 LSTM, 선형 RNN, 패널티 없는 ReLU, 질문만 모델을 기준선으로 둠");
}

// ------------------------------------------------------------------------------------------------------------------ 3 design
{
  const s = pres.addSlide();
  frame(s, "02", "연구 방법", "실험 설계와 진행 현황", "팀이 정한 초기 설정으로 맞췄습니다. 입력은 영상 25프레임과 자연어 질문이며 서술형만 사용합니다.");
  box(s, 0.6, 1.8, 5.4, 4.9, "고정한 것 (팀 설정)", [
    "영상 3000개: 학습 0-1999 / 검증 2000-2499 / 평가 2500-2999",
    "서술형 질문만 (21개 답 중 하나)",
    "입력: 영상당 25프레임 + 자연어 질문",
    "뉴런 100개, 60 에포크, 같은 데이터·분할로 비교",
    "이동은 첫 에포크부터 (open_epoch 1)"], { size: 14, hsize: 22, hcolor: CORAL });
  s.addText("실험요인 진행현황 - 동적 좌표", { x: 6.4, y: 1.8, w: 6.3, h: 0.4, fontSize: 18, bold: true, fontFace: FONT, margin: 0, isTextBox: true });
  table(s, [["요인", "바꾸는 값", "상태"],
    ["이동 방식", "고정 / position / swap / both / relayout", "완료 (시드 2)"],
    ["비교 모델", "LSTM / 선형 RNN / 패널티 없음 / 질문만", "완료 (시드 2)"],
    ["입력 병목 진단", "질문만, 유형별 정확도, 탐침", "완료"],
    ["구조 마스크", "A / B2 / B1 + 이동 (주석 입력)", "완료"],
    ["영상 특징 개선", "물체 단위 인식, 데이터 확대", "예정"]], { x: 6.4, y: 2.3, w: 6.3, colW: [1.5, 3.4, 1.4], rowH: 0.65, fontSize: 12 });
  foot(s, "시드 수는 현재 2개이며 3개로 늘릴 예정. 구조 마스크 비교는 영상 입력의 정확도 천장 때문에 주석 입력(진단용)으로 수행");
}

// ------------------------------------------------------------------------------------------------------------------ 4 dynamic result
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "동적 이동 결과 - 팀 설정", "이동 방식에 따른 정확도 차이는 없습니다. 모든 모델이 0.38-0.40에 있고 질문만 보는 모델도 0.35입니다.");
  const rows = [["방식 / 모델", "평가 정확도", "평균 이동", "배선 비용", "파라미터"]];
  D.dyn_team.forEach((r) => rows.push([r.name, { t: pm(r.acc) + (nOf(r.acc) < 2 ? "" : ""), hi: r.name === "질문만 (영상 끔)" }, r.move === null ? "-" : f2(r.move), r.name === "질문만 (영상 끔)" ? "-" : f0(r.wire), r.params ? r.params.toLocaleString("en-US") : "-"]));
  table(s, rows, { x: 0.6, y: 1.8, w: 7.9, colW: [2.4, 1.9, 1.1, 1.2, 1.3], rowH: 0.46, fontSize: 13 });
  box(s, 8.8, 1.8, 3.95, 4.9, "읽는 법", [
    "고정, position, swap, both, relayout의 정확도는 0.388-0.395로 차이 없음",
    "LSTM 0.397도 같은 구간",
    "질문만 보는 모델이 0.350 - 영상은 약 +4점만 보탬",
    "배선 비용은 position, both가 고정보다 낮음 (이동이 연결 비용을 줄임)"], { size: 13, hsize: 20, hcolor: CORAL });
  foot(s, "평가 영상 2500-2999, 서술형, ResNet18 + 단어 묶음, 시드 평균 ± 표준편차(질문만은 시드 1개). 평균 이동은 격자 간격 1 기준");
}

// ------------------------------------------------------------------------------------------------------------------ 5 movement figure
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "이동 후 뉴런 배치", "position과 both는 뉴런이 한쪽에 덩어리로 모입니다. swap과 relayout은 격자 안에서 자리만 바꿉니다. 진할수록 많이 이동한 뉴런입니다.");
  s.addImage({ path: "results/figs/compare_team_e28a_compact.png", x: 0.55, y: 1.7, w: 12.2, h: 5.1, objectName: "movement comparison" });
  foot(s, "시드 1, 팀 설정(서술형, ResNet18). position, both는 가장 가까운 쌍이 0.42-0.44로 최소거리 0.5 미만 (반발이 벌점이라 엄격하지 않음)");
}

// ------------------------------------------------------------------------------------------------------------------ 6 modules
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "모듈과 스몰월드는 어디서 생기나", "이동만으로는 모듈이 안 생기고, 구조 마스크(B2, B1)를 주면 이동으로 모듈이 공간에서 갈라지고 모듈 사이는 게이트웨이로만 연결됩니다.");
  s.addImage({ path: "results/figs/modules_e30_structures_compact.png", x: 0.55, y: 1.75, w: 8.2, h: 8.2 * 4.6 / 8.4, objectName: "module structures" });
  const e = Object.fromEntries(D.e30.map((r) => [r.name, r]));
  const row = (label, k) => [label, f2(e[k] && e[k].Q), f2(e[k] && e[k].sigma), f0(e[k] && e[k].edges), f3(e[k] && e[k].acc_all[0])];
  table(s, [["구조 + 이동", "모듈성 Q", "σ", "활성 연결", "정확도"],
    row("없음 + position", "position_full"), row("B2 + position", "position_B2"), row("B1 + position", "position_B1"), row("B1 + swap", "swap_B1")],
    { x: 8.85, y: 1.8, w: 4.0, colW: [1.35, 0.7, 0.5, 0.75, 0.7], rowH: 0.55, fontSize: 11 });
  s.addText([
    { text: "구조 마스크 없이 이동만으로는 모듈이 안 생김 (Q 0.2)", options: { bullet: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: "B2, B1 + position: 모듈이 공간에서 분리되고 모듈 사이는 가는 연결로만 이어짐", options: { bullet: true, breakLine: true, paraSpaceAfter: 4 } },
    { text: "swap은 격자를 유지해 모양은 크게 안 바뀜", options: { bullet: true } }],
    { x: 8.85, y: 4.75, w: 4.0, h: 1.9, fontSize: 13, color: INK, fontFace: FONT, valign: "top", margin: 0, isTextBox: true });
  foot(s, "주석 입력(진단용), 4유형, 학습 0-1999. 그림은 시드 1, 표는 시드 평균. Q = 모듈성, σ = small-world 계수, 색 = 모듈");
}

// ------------------------------------------------------------------------------------------------------------------ 7 CNN bottleneck
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "정확도가 0.4에 머무는 이유 - 영상 특징", "영상이 도움이 되는 건 색 질문뿐이고, 개수와 존재 여부는 영상 효과가 없습니다. 고정 ResNet 특징에 충돌과 개수 정보가 거의 없기 때문입니다.");
  const mj = D.majority && D.majority.test && D.majority.test.majority_per_group;
  const sub = [["개수", "desc_count"], ["존재 여부", "desc_exist"], ["색", "desc_query_color"], ["모양", "desc_query_shape"], ["재질", "desc_query_material"]];
  const rows = [["서술형 유형", "최빈값", "질문만", "영상 (고정 seRNN)", "영상 (LSTM)"]];
  const get = (i, k) => D.cnn_subtype[i][k];
  sub.forEach(([n, k]) => rows.push([n, mj ? f3(mj[k]) : "-", f3(get(0, k)), { t: f3(get(1, k)), hi: k === "desc_query_color" }, f3(get(2, k))]));
  rows.push(["전체", mj ? f3(mj.acc_all) : "-", f3(D.cnn_subtype[0].all[0]), f3(D.cnn_subtype[1].all[0]), f3(D.cnn_subtype[2].all[0])]);
  table(s, rows, { x: 0.6, y: 1.8, w: 7.6, colW: [1.5, 1.2, 1.2, 2.0, 1.7], rowH: 0.5, fontSize: 13 });
  box(s, 8.5, 1.8, 4.25, 3.1, "탐침으로 읽히는 것", [
    "색, 모양, 재질: 0.88-0.92 (읽힘)",
    "물체 개수: 0.70 대 찍기 0.37 (약하게)",
    "충돌: 0.48 대 0.49 (안 읽힘)"], { size: 13, hsize: 16, hcolor: CORAL });
  box(s, 8.5, 5.05, 4.25, 1.65, "결론", ["영상은 색에서만 +17점, 개수·존재는 +0", "병목은 영상 특징(입력)"], { size: 13, hsize: 16, hcolor: CORAL });
  foot(s, "ResNet18 + 단어 묶음, 팀 설정(서술형), 시드 평균(질문만 1개, 영상 2개). 최빈값 = 유형별로 가장 흔한 답을 고를 때의 정확도");
}

// ------------------------------------------------------------------------------------------------------------------ 8 annotation diagnostic (e30)
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "입력 병목을 없애면 (주석 입력, 진단용)", "영상 특징 대신 물체 정보(주석)를 넣어서, 정확도 천장이 올라갔을 때 이동, 구조, 모델 종류가 정확도를 바꾸는지 봤습니다. 논문 비교용 숫자는 아닙니다.");
  const names = [["고정", "fixed_full"], ["position", "position_full"], ["swap", "swap_full"], ["relayout", "relayout_full"], ["고정 + A", "fixed_A"], ["고정 + B2", "fixed_B2"], ["고정 + B1", "fixed_B1"],
    ["position + B2", "position_B2"], ["position + B1", "position_B1"], ["swap + B1", "swap_B1"], ["LSTM", "lstm"], ["선형 RNN", "linear_rnn"], ["패널티 없는 ReLU", "relu_nopen"], ["질문만", "question_only"]];
  const e = Object.fromEntries(D.e30.map((r) => [r.name, r]));
  const rows = [["모델", "서술형", "설명(문항)", "예측(문항)", "반사실(문항)", "전체", "시드"]];
  names.forEach(([n, k]) => { const r = e[k]; if (r) rows.push([n, pm(r.descriptive_per_option), pm(r.explanatory_per_question), pm(r.predictive_per_question), pm(r.counterfactual_per_question), pm(r.acc_all), String(r.n)]); });
  table(s, rows, { x: 0.6, y: 1.75, w: 8.9, colW: [1.75, 1.3, 1.3, 1.3, 1.35, 1.3, 0.6], rowH: 0.33, fontSize: 10.5 });
  box(s, 9.75, 1.75, 3.0, 5.05, "읽는 법", [
    "고정, position, swap, relayout: 전체 0.520-0.524로 차이 없음",
    "구조 마스크: 활성 연결 약 1/10 (1535개에서 115-158개), 모듈성 Q 0.2에서 0.6-0.8, 정확도 0.510-0.513 (약 1점 낮음)",
    "패널티 있는 seRNN 0.521이 패널티 없는 ReLU 0.486, 선형 0.477, LSTM 0.506보다 높음",
    "질문만 0.436: 주석 입력이 +9점"], { size: 12, hsize: 18, hcolor: CORAL });
  foot(s, "학습 영상 2000개 기준(전체 0.5 안팎). 학습 9000개 기준 0.707은 다음 장. 평가 2500-2999, 시드 평균 ± 표준편차, 문항 = 객관식 문제 전체 정답");
}

// ------------------------------------------------------------------------------------------------------------------ 9 data / question encoding
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "데이터 양과 질문 인코딩 효과", "같은 주석 입력에서 학습 영상을 2000개에서 15000개로 늘리고 질문 인코딩을 바꾼 결과입니다. 평균 벡터는 질문의 구조를 지워 객관식이 약합니다.");
  const sc = Object.fromEntries(D.scale.map((r) => [r.exp + ":" + r.name, r]));
  const line = (label, k) => { const r = sc[k]; return [label, pm(r.acc_all), pm(r.descriptive_per_option), pm(r.explanatory_per_question), pm(r.predictive_per_question), pm(r.counterfactual_per_question)]; };
  table(s, [["주석 입력 (진단용)", "전체", "서술형", "설명(문항)", "예측(문항)", "반사실(문항)"],
    line("2000개 - 평균 벡터", "e24_controls:ann_w2v"), line("2000개 - program 토큰", "e24_controls:ann_prog"),
    line("15000개 - 평균 벡터", "e25_scale:ann_mean"), line("15000개 - 단어 순서", "e25_scale:ann_pos"),
    line("15000개 - program 토큰", "e25_scale:ann_prog"), line("15000개 - 코어가 단어를 읽음", "e25_scale:ann_seqcore")],
    { x: 0.6, y: 1.75, w: 12.15, colW: [3.5, 1.7, 1.7, 1.75, 1.75, 1.75], rowH: 0.5, fontSize: 13 });
  box(s, 0.6, 5.5, 12.15, 1.3, "읽는 법", [
    "같은 입력에서 질문 인코딩만 바꿔도 전체 0.566-0.707 (학습 영상 9000개). program 기준 학습 2000개 0.521 대 9000개 0.707",
    "program = 질문을 푸는 단계(예: 물체 고르기, 충돌 찾기, 상대 구하기, 모양 묻기)의 횟수. 정답이 아닌 진단용"], { size: 12, hsize: 14, hcolor: CORAL });
  foot(s, "평가 영상 10000-14999(15000개 실험) 또는 2500-2999(2000개 실험). 시드 3개 평균 (코어가 단어를 읽음은 시드 1개). 단어 순서 인코더에는 LSTM을 쓰지 않음");
}

// ------------------------------------------------------------------------------------------------------------------ 10 diagnosis & directions
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "병목 진단과 해결 방향", "증거가 있는 것만 적었습니다. 방향은 제 해석이며 팀과 정해야 합니다.");
  table(s, [["문제", "증거", "해결 방향"],
    ["영상 특징에 충돌, 개수 정보가 없음", "질문만 0.350 대 영상 0.39 / 탐침 충돌 0.48 대 0.49", "물체 단위 인식 (검출, 추적)"],
    ["질문을 평균 벡터로 압축", "주석 입력: 평균 0.566, 단어 순서 0.650, program 0.707", "순서를 살리는 질문 인코딩 (순환 없이)"],
    ["학습 데이터가 적음", "주석 + program: 2000개 0.521 대 15000개 0.707", "영상 특징을 더 많은 영상으로 확대"],
    ["이동, 구조가 정확도를 바꾸지 않음", "영상 0.38-0.40, 주석 0.51-0.52 모두 차이 없음 (구조는 연결 수 1/10)", "정확도 대신 효율(연결 수, 모듈성)로 평가하고, 천장을 올린 뒤 재비교"]],
    { x: 0.6, y: 1.8, w: 12.15, colW: [3.5, 5.0, 3.65], rowH: 0.95, fontSize: 14 });
  foot(s, "주석 입력은 진단용이며 논문(주석을 입력으로 쓰지 않음)과의 비교 기준이 아님");
}

// ------------------------------------------------------------------------------------------------------------------ 11 decisions
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "팀과 정해야 할 것", "아래 항목은 팀 결정이 필요합니다.");
  box(s, 0.6, 1.8, 12.15, 4.9, "논의 안건", [
    "기준 입력: 영상(25프레임) 확정, 주석은 진단용으로만 사용",
    "영상 특징: ResNet18 대 ResNet-50, 그리고 물체 단위 인식 도입 여부 (팀 TODO ImagePerception)",
    "질문 인코딩 확인: 팀 코드는 program 개수, 팀 메시지는 '자연어 질문' - 어느 쪽인가? (논문 기준 모델은 단어 평균 또는 LSTM)",
    "분할: 학습 0-1999 / 검증 2000-2499 / 평가 2500-2999 유지 여부 (정적 실험은 0-9999 / 10000-14999)",
    "평가 범위: 서술형만, 4유형 전부 (객관식은 입력 병목이 풀려야 의미)"], { size: 18, hsize: 22, hcolor: CORAL });
}

// ------------------------------------------------------------------------------------------------------------------ 12 AI use
{
  const s = pres.addSlide();
  frame(s, "03", "중간 결과", "AI(Claude) 활용 내용", "수정이 필요하면 직접 고치세요.");
  box(s, 0.6, 1.8, 12.15, 4.9, "AI 활용", [
    "실험 코드 작성·수정: 동적 좌표(position, swap, relayout) 실행 옵션, 주석 입력, 질문 인코딩 옵션 추가",
    "분석 스크립트: 영상 특징 탐침, 유형별 정확도 표, 구조 지표(모듈성 Q, σ)",
    "시각화 코드: 이동 전후 배치 그림, 모듈 구조 그림",
    "발표자료 초안 작성 (이 자료), 발표 스크립트 초안은 슬라이드 노트에 있음",
    "결과 해석은 초안이며 팀원의 검토를 거쳐 수정"], { size: 16, hsize: 22, hcolor: CORAL });
}

pres.writeFile({ fileName: "dynamic_seRNN_slides.pptx" }).then((f) => console.log("saved", f));
