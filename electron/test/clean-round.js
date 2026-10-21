// 一整局**干净的**模拟牌局 —— 从发牌到流局，逐手检查手牌跟踪。
//
// ## 为什么要有这个文件
//
// 之前我一直拿 `build/ws-frames` 里的真实录像当判据，而那段录像有断点
//（一局中间夹着 ActionMJStart、缺帧、把不同场次接在一起）。于是测试报出来的
// 东西大半是录像自己的毛病，我拿着这些数字来回改，越改越糊涂，真 bug 反而漏了 ——
// 漏掉的那个就是「东家上来就失准」：
//
//   庄家起手 14 张、而且他**第一个打牌**，那时候还没有任何「带牌名的摸牌」
//   可以依据，mySeat 还是 null -> 那一手没被认成我的 -> 手牌停在 14
//   -> 下一次摸牌被保险丝拦掉 -> 之后整局全废
//
// 这段录像里恰好看不出来（它的局边界是乱的）。所以现在补一个**合成**的干净牌局：
// 每一步都由测试自己发，状态完全可控，错了就是代码错。
const { GameState } = require('../lib/gamestate');

const SEAT = ['东', '南', '西', '北'];
const checks = [];
const check = (name, ok, detail = '') => checks.push([name, ok, detail]);

/** 造一局：myHand 是发牌，meSeat 是「我」坐哪（庄家 = 0），按正常顺序摸打 rounds 巡 */
function playRound({ meSeat, players, hand, rounds = 6 }) {
  const g = new GameState();
  g.onAction('ActionMJStart', Buffer.alloc(0));
  g._newRound({
    hand, dora: ['1p'], chang: 0, ju: 0, ben: 0, leftTileCount: 69,
  });
  const startLen = g.hand.length;
  const problems = [];

  const seen = (tag) => {
    const eff = g.hand.length + 3 * g.meldCount;
    if (g.mySeat !== null && eff !== 13 && eff !== 14) {
      problems.push(`${tag}: 等效 ${eff}（手牌 ${g.hand.length} + 副露 ${g.meldCount}）`);
    }
  };

  // 庄家先打（庄家起手 14 张）；其余人依次摸打
  const order = [];
  for (let k = 0; k < players; k++) order.push(k);

  // 第一手：庄家打牌
  if (g.dealer === null || g.hand.length >= 14) {
    // 庄家就是座位 0；如果我坐 0 那我起手 14 张
    const dealerSeat = 0;
    const tile = g.hand[g.hand.length - 1];
    g._discard({ seat: dealerSeat, tile, isLiqi: false, moqie: false });
    seen(`庄家(${SEAT[dealerSeat]})打第一张`);
  }

  for (let turn = 0; turn < rounds; turn++) {
    for (const seat of order) {
      if (seat === 0 && turn === 0) continue;          // 庄家第一手已经打过
      // 摸牌：只有我自己的那条带牌名
      const mine = seat === meSeat;
      const tile = mine ? g.hand[g.hand.length - 1] : null;   // 占位，随便给一张
      g._deal({ seat, tile: mine ? pickDraw(g) : null, leftTileCount: 60 - turn });
      seen(`${SEAT[seat]}摸牌`);
      // 打牌
      const t = mine ? g.hand[g.hand.length - 1] : '1z';
      g._discard({ seat, tile: t, isLiqi: false, moqie: false });
      seen(`${SEAT[seat]}打牌`);
    }
  }
  return { g, problems, startLen };
}

/** 给一张「手里还没有的牌」当摸牌用，避免和手牌计数打架 */
let pool = 0;
function pickDraw(g) {
  const all = ['1m', '9m', '1p', '2p', '3p', '4p', '5p', '6p', '7p', '8p', '9p',
    '1s', '2s', '3s', '4s', '5s', '6s', '7s', '8s', '9s', '1z', '2z', '3z', '4z', '5z', '6z', '7z'];
  for (let i = 0; i < all.length; i++) {
    const t = all[(pool + i) % all.length];
    if (!g.hand.includes(t)) { pool = (pool + i + 1) % all.length; return t; }
  }
  return all[pool++ % all.length];
}

// ---- 三麻：我是庄家 ----
{
  const r = playRound({
    meSeat: 0, players: 3,
    hand: ['1m', '9m', '2p', '3p', '4p', '5s', '6s', '7s', '1z', '1z', '4z', '5z', '6z', '7z'],
  });
  check('三麻·我是庄家：整局没有失准', r.problems.length === 0 && !r.g.handSuspect,
    r.problems[0] || `最终等效 ${r.g.hand.length + 3 * r.g.meldCount}`);
  check('  庄家认出来了，而且就是我', r.g.dealer === 0 && r.g.mySeat === 0,
    `dealer=${r.g.dealer} mySeat=${r.g.mySeat}`);
}

// ---- 三麻：我是南家（非庄家，13 张起手）----
{
  const r = playRound({
    meSeat: 1, players: 3,
    hand: ['2m', '3m', '2p', '3p', '4p', '5s', '6s', '7s', '1z', '1z', '4z', '5z', '6z'],
  });
  check('三麻·我是南家：整局没有失准', r.problems.length === 0 && !r.g.handSuspect,
    r.problems[0] || `最终等效 ${r.g.hand.length + 3 * r.g.meldCount}`);
}

// ---- 四麻：我是庄家 ----
{
  const r = playRound({
    meSeat: 0, players: 4,
    hand: ['1m', '9m', '2p', '3p', '4p', '5s', '6s', '7s', '1z', '1z', '4z', '5z', '6z', '7z'],
  });
  check('四麻·我是庄家：整局没有失准', r.problems.length === 0 && !r.g.handSuspect,
    r.problems[0] || '');
}

// ---- 吃牌：手牌必须正好扣掉「自己出的两张」，不能多也不能少 ----
//
// 这是用户报的「吃牌后大概率失焦」的回归。真实录像里 ActionChiPengGang 的
// `tiles` 顺序是 `["4s","6s","5s"]`（被叫的 5s 在最后），而 `froms` 以前因为
// packed 编码没解出来、是空数组 —— 老代码于是猜错扣哪张，等效张数冲到 15，
// 接着手牌跟踪整局冻结（界面上就是「失准」，用户叫它「失焦」）。
for (const froms of [[1, 1, 0], []]) {
  const g = new GameState();
  g.onAction('ActionMJStart', Buffer.alloc(0));
  g._newRound({
    hand: ['1m', '9m', '2p', '3p', '4p', '5p', '6p', '7p', '4s', '6s', '1z', '1z', '2z'],
    dora: ['3z'], chang: 0, ju: 0, ben: 0, leftTileCount: 60,
  });
  g.mySeat = 1;
  g._discard({ seat: 0, tile: '5s', isLiqi: false, moqie: false });   // 上家打 5s
  const r = g._meld({ seat: 1, type: 0, tiles: ['4s', '6s', '5s'], froms });
  const tag = froms.length ? 'froms 有' : 'froms 缺（老录像）';
  check(`吃牌（${tag}）：扣掉手牌里的 4s/6s，不把被叫的 5s 扣进手牌`,
    !r.rolledBack && !g.handSuspect && g.melds.length === 1
      && !g.hand.includes('4s') && !g.hand.includes('6s') && !g.hand.includes('5s')
      && g.hand.length === 11 && g.effective === 14,
    `rolledBack=${!!r.rolledBack} suspect=${g.handSuspect} 手牌 ${g.hand.join(' ')} ` +
    `等效 ${g.effective}（期望 11 张 / 14 等效）`);
  if (!r.rolledBack) {
    g._discard({ seat: 1, tile: '2z', isLiqi: false, moqie: false });
    check(`  吃完打一张：回到 10 张 + 1 副露 = 13 等效（${tag}）`,
      g.hand.length === 10 && g.effective === 13 && !g.handSuspect,
      `${g.hand.length} 张 + ${g.meldCount} 副露 = ${g.effective}`);
  }
}

// ---- 拔北：净 0 张，不该失准 ----
{
  const g = new GameState();
  g.onAction('ActionMJStart', Buffer.alloc(0));
  g._newRound({ hand: ['1m', '9m', '2p', '3p', '4p', '5s', '6s', '7s', '1z', '1z', '4z', '5z', '6z'],
                dora: ['1p'], chang: 0, ju: 0, ben: 0, leftTileCount: 69 });
  g._discard({ seat: 0, tile: '1m', isLiqi: false, moqie: false });   // 庄家先打
  g._deal({ seat: 1, tile: null }); g._discard({ seat: 1, tile: '1z' });
  g._deal({ seat: 2, tile: null }); g._discard({ seat: 2, tile: '2z' });
  const before = g.hand.length;
  g._deal({ seat: 0, tile: '4z' });                    // 我摸到北
  const afterDraw = g.hand.length;
  g._kita({ seat: 0 });                                // 拔北：扣掉北
  const afterKita = g.hand.length;
  g._deal({ seat: 0, tile: '2s' });                    // 补摸（服务器紧接着发）
  check('拔北：扣一张 + 补一张 = 净 0', afterDraw - afterKita === 1 && g.hand.length === afterDraw,
    `${before} -> 摸 ${afterDraw} -> 拔 ${afterKita} -> 补 ${g.hand.length}`);
  check('  拔北之后没有失准', !g.handSuspect, '');
}

// ---- 自风必须每小局跟着庄家轮转 ----
//
// 踩过的坑：界面上拿**座位号**当风显示，于是只有第一小局对（那时庄家坐 0 位），
// 之后每局都显示错的风。用户原话：「场上各家对应的风还是只有第一小局能自动判断」。
{
    const g = new GameState();
    const seen = [];
    for (let ju = 0; ju < 4; ju++) {
        g._newRound({ hand: ['1m', '9m', '2p', '3p', '4p', '5s', '6s', '7s', '1z', '1z', '4z', '5z', '6z'],
                      dora: ['1p'], chang: 0, ju, ben: 0, leftTileCount: 69 });
        g.mySeat = 0;
        const dealerSeat = ju % 3;
        g._discard({ seat: dealerSeat, tile: '1z', isLiqi: false, moqie: false });  // 第一张牌定庄家
        seen.push({ ju, dealer: g.dealer, wind: g.myWind });
    }
    check('庄家每小局都重新认出来（不是一直第一局那个）',
        seen.map((x) => x.dealer).join(',') === '0,1,2,0',
        seen.map((x) => x.dealer).join(','));
    check('我的自风每小局跟着轮转（东→西→南→东）',
        seen.map((x) => x.wind).join(',') === '东,西,南,东',
        seen.map((x) => x.wind).join(','));
    check('  自风不再等于座位号（第 2 小局就不同了）',
        seen[1].wind !== '东', `第 2 小局我坐 0 位，自风是 ${seen[1].wind}`);
}

let fail = 0;
for (const [name, ok, detail] of checks) {
  if (!ok) fail++;
  console.log(`${ok ? '✅' : '❌'} ${name}${detail ? '　' + detail : ''}`);
}
console.log();
console.log(fail === 0 ? `✅ 干净牌局 ${checks.length} 项全部通过`
  : `❌ ${fail} 项不通过`);
process.exit(fail === 0 ? 0 : 1);
