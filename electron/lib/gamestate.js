// 牌局状态 —— **全部来自协议**，一点识图都不用。
//
// 数据来源（字段号照抄 .refs/mahjongsoul.proto，解码见 lib/actions.js）：
//
//   ActionNewRound      手牌（字段 4）、宝牌（14）、场况
//   ActionDealTile      摸牌（字段 2）—— **只有自己摸的那张带牌名**
//   ActionDiscardTile   座位 / 牌 / 立直 / 摸切 / 振听
//   ActionChiPengGang   副露的**具体牌**（字段 3）+ 类型 + 来源
//   ActionBaBei         拔北（三麻特有）
//   ActionAnGangAddGang 暗杠 / 加杠
//
// 一句话：手牌、宝牌、副露、牌河、立直，**全都读得到**。
// 以前那套「识图 + 开局认一次」是白做的 —— 原因写在 lib/liqi.js 开头的注释里。
const { decodeAction } = require('./actions');
const { sortHand } = require('./rowgeom');

const SEAT_NAMES = ['东', '南', '西', '北'];

class GameState {
  constructor() {
    this.handNo = 0;
    this.scores = [];
    this.lastDraw = null;
    this.reset();
  }

  reset() {
    /** 我的手牌（含刚摸到还没打的那张），顺序按雀魂显示顺序 */
    this.hand = [];
    /** 我的副露：[{ type: 'chi'|'pon'|'kan'|'kita', tiles, froms }] */
    this.melds = [];
    /** **四家**的副露（按座位）。模型要看别人副露了几组才判断得出威胁，
     *  只给自己的那份等于让模型以为对家全是门清。 */
    this.meldsBySeat = [[], [], [], []];
    /** 宝牌指示牌 */
    this.dora = [];
    /** 四家牌河 */
    this.discards = [[], [], [], []];
    /** 哪几家立直了（一局一次，不可撤销） */
    this.riichi = [false, false, false, false];
    /** 各家摸切了几张 */
    this.moqie = [0, 0, 0, 0];
    /** 各家拔了几个北（三麻特有）。模型要用它算「北还剩几张」——
     *  写死 0 的话模型看到的北的数量是错的。 */
    this.kita = [0, 0, 0, 0];
    /** 各家是否处于「一发」中：立直之后到下一次打牌之前。
     *  协议里没有直接字段，从立直和打牌推出来。模型用它判断「现在最该防谁」。 */
    this.ippatsu = [false, false, false, false];
    /**
     * **推断出来的**座位 —— 只用它跟踪手牌和牌河。
     *
     * 来源是协议本身：带牌名的 `ActionDealTile` 就是服务器发给**本人**的
     *（别人摸牌不会把牌名发给你）。所以这条推断跟界面上的手动设置**无关**。
     */
    this.mySeat = null;
    /**
     * **手动**设的座位（界面上点「我坐」那一行）。
     *
     * 只影响两件事：显示的自风、喂给模型的自风。
     * **绝不影响手牌跟踪** —— 手牌是「我手里有什么牌」，跟「我叫什么风」无关。
     *
     * 踩过的坑：原来手牌跟踪也看这个值，于是用户点一下「换座位」，
     * 下一条摸牌就被判成「不是我的」而丢掉，手牌少一张，接着就失准。
     * 用户的原话是「理论上手牌数据和切换自风应该无关啊」—— 他是对的。
     */
    this.seatOverride = null;
    /**
     * 座位投票：{座位: 票数}，每局清空。
     *
     * 为什么不「见到就信」：一局里带牌名的摸牌**只应该来自我自己**，
     * 但录像有断点、也可能混着不同场次，于是会冒出别的座位的票。
     * 直接采信就会来回横跳（实测过：座位在东南西北之间反复纠正，
     * 手牌因此少了一张）。取**这一局里票最多的那个**最稳。
     */
    this._seatVotes = {};
    /** 这一桌是几麻。三麻没有 2m~8m，牌河一出现就是四麻 */
    this.sanma = true;
    this.chang = 0; this.ju = 0; this.ben = 0;
    this.leftTileCount = 0;
    /**
     * 这一局的庄家坐哪。
     *
     * **不用猜**：庄家一定打这一局的第一张牌（麻将规则）。所以拿本局第一条
     * `ActionDiscardTile` 的座位就是庄家 —— 这是硬信息，比任何推断都可靠。
     * 自风 = 离庄家差几家（aimodel 里也是这么算的，见 engine.seat_wind）。
     */
    this.dealer = null;
    /** 刚摸牌的是哪一家（用来验证「摸牌的人下一手必须打牌」） */
    this._pendingDrawSeat = null;
    this.lastAction = null;
    this.handSuspect = false;      // 新的一局：发牌消息是完整的，重新开始信
    this.desync = 0;
    this.updatedAt = Date.now();
  }

  /**
   * 手牌 + 3×副露 = 等效张数；14（含刚摸的）就说明轮到我打牌。
   *
   * **拔北不算副露**：它和杠一样是「拿掉一张、再补摸一张」，
   * 手牌张数不变（仍然要凑 13/14 张形）。算进去的话 14 + 3×拔北数
   * 会变成 17、20 这种数字，整手牌就被判成「不合法」而完全不给建议 ——
   * 实测在雀魂里拔两次北就复现了。
   */
  get meldCount() { return this.melds.filter((m) => m.type !== 'kita').length; }
  get effective() { return this.hand.length + 3 * this.meldCount; }

  /**
   * 手动设座位（界面点的那一下）。
   *
   * **只写 seatOverride**，不碰 this.mySeat、更不碰手牌。
   * 见 seatOverride 的注释。
   */
  setMySeat(seat) {
    if (seat < 0 || seat > 3) return this.effectiveSeat;
    this.seatOverride = seat;
    this._touch();
    return this.effectiveSeat;
  }

  /** 界面上/模型里应该用的座位：手动设过就用手动那个，否则用推断的 */
  get effectiveSeat() {
    return this.seatOverride !== null ? this.seatOverride : this.mySeat;
  }

  /** 我是几号座位（用于牌河这类按座位索引的东西） */
  get _seatIndex() {
    const n = this.sanma ? 3 : 4;
    const s = this.effectiveSeat;
    return (s === null || s === undefined) ? null : (s % n);
  }

  /**
   * **我的自风**（东/南/西/北）—— 注意这**不是座位号**。
   *
   * 自风 = 离庄家差几家（和 aimodel 的 engine.seat_wind 一致，三麻只有东南西）。
   * 庄家每小局轮转，所以自风每小局都会变。
   *
   * 踩过的坑：界面上一直拿**座位号**当风显示（`SEAT_NAMES[mySeat]`），
   * 于是只有第一小局是对的（那时庄家恰好坐 0 位），之后每一局都显示错的风 ——
   * 用户的原话是「场上各家对应的风还是只有第一小局能自动判断」。
   */
  get myWind() {
    const n = this.sanma ? 3 : 4;
    const s = this.effectiveSeat;
    if (s === null || s === undefined || this.dealer === null) return null;
    const table = n === 3 ? ['东', '南', '西'] : ['东', '南', '西', '北'];
    return table[(((s - this.dealer) % n) + n) % n];
  }

  /** 各家坐在哪个风上（按**座位号**索引）—— 桌上显示用 */
  get windsBySeat() {
    const n = this.sanma ? 3 : 4;
    const table = n === 3 ? ['东', '南', '西'] : ['东', '南', '西', '北'];
    const out = [null, null, null, null];
    if (this.dealer === null) return out;
    for (let i = 0; i < n; i++) {
      out[i] = table[(((i - this.dealer) % n) + n) % n];
    }
    return out;
  }

  snapshot() {
    return {
      hand: this.hand.slice(),
      melds: this.melds.map((m) => ({ ...m, tiles: m.tiles.slice() })),
      meldsBySeat: this.meldsBySeat.map((row) => row.map((m) => ({ ...m, tiles: m.tiles.slice() }))),
      dora: this.dora.slice(),
      discards: this.discards.map((r) => r.slice()),
      riichi: this.riichi.slice(),
      moqie: this.moqie.slice(),
      kita: this.kita.slice(),
      ippatsu: this.ippatsu.slice(),
      mySeat: this.effectiveSeat,          // 界面显示用（手动设过就是手动那个）
      inferredSeat: this.mySeat,            // 协议推出来的（排查用）
      seatOverride: this.seatOverride,
      myWind: this.myWind,                 // 我的自风（东/南/西/北）—— 不是座位号
      windsBySeat: this.windsBySeat,       // 各家坐在哪个风
      players: this.sanma ? 3 : 4,
      sanma: this.sanma,
      handNo: this.handNo,
      desync: this.desync || 0,
      handSuspect: this.handSuspect,
      dealer: this.dealer,
      effective: this.effective,
      meldCount: this.meldCount,
      lastDraw: this.lastDraw,
      scores: this.scores.slice(),
      leftTileCount: this.leftTileCount,
      chang: this.chang,      // 场风（0=东场）—— 网络要用它算自风
      ju: this.ju,
      ben: this.ben,
      updatedAt: this.updatedAt,
    };
  }

  _touch() {
    this.updatedAt = Date.now();
    if (this.onChange) {
      try { this.onChange(this.snapshot()); } catch (e) { /* 回调出错不影响状态 */ }
    }
  }

  /**
   * 标记「手牌已经不可信」。
   *
   * 一旦手牌和牌河对不上（漏了动作、中途重连、录像断点），后面**怎么改都是错的** ——
   * 继续改只会越错越远（曾经涨到 16 张等效）。所以这里立个旗子：
   * 之后不再动手牌，界面也会如实说「仅供参考」。
   * 下一局发牌时自动清掉（发牌消息是完整的、可信的）。
   */
  _markDesync(why) {
    if (!this.handSuspect) {
      this.handSuspect = true;
      console.log(`[game] ⚠️ 手牌跟踪失准：${why}　` +
        '之后不再修改手牌，等下一局发牌重置（这段的建议仅供参考）');
    }
    this.desync = (this.desync || 0) + 1;
  }

  /** 从手牌里拿掉 count 张 tile，返回实际拿掉几张 */
  _takeFromHand(tile, count = 1) {
    let n = 0;
    for (let i = this.hand.length - 1; i >= 0 && n < count; i--) {
      if (this.hand[i] === tile) { this.hand.splice(i, 1); n++; }
    }
    return n;
  }

  /** 处理一条 ActionPrototype，返回给界面/日志用的事件描述 */
  onAction(action, payload) {
    const d = decodeAction(action, payload);
    if (!d) return null;
    this.lastAction = action;

    switch (action) {
      case 'ActionNewRound': return this._newRound(d);
      case 'ActionDealTile': return this._deal(d);
      case 'ActionDiscardTile': return this._discard(d);
      case 'ActionChiPengGang': return this._meld(d);
      case 'ActionBaBei': return this._kita(d);
      case 'ActionAnGangAddGang': return this._kan(d);
      case 'ActionHule':
        if (d.scores && d.scores.length) this.scores = d.scores.slice();
        this._touch();
        return { kind: 'hule', scores: this.scores.slice() };
      case 'ActionMJStart':
        this.reset();
        this._touch();
        return { kind: 'gamestart' };
      case 'ActionGameEnd':
      case 'ActionNoTile':
        this._touch();
        return { kind: 'gameend' };
      default:
        return null;
    }
  }

  // ---- 各动作 -------------------------------------------------------------

  _newRound(d) {
    this.handNo += 1;
    this.hand = sortHand(d.hand || []);
    this.dora = (d.dora || []).slice();
    this.chang = d.chang; this.ju = d.ju; this.ben = d.ben;
    this.leftTileCount = d.leftTileCount || 0;
    this.discards = [[], [], [], []];
    this.riichi = [false, false, false, false];
    this.moqie = [0, 0, 0, 0];
    this.kita = [0, 0, 0, 0];
    this.ippatsu = [false, false, false, false];
    this.melds = [];
    this.meldsBySeat = [[], [], [], []];
    this.lastDraw = null;
    this._seatVotes = {};        // 座位每局重新投票（跟手动设的自风无关，见 seatOverride）

    // ---- 这几项**必须每小局都重置** ----
    //
    // 踩过的坑：这些原来只在 `reset()`（= ActionMJStart，整场开始）里清，
    // 而**不是**在这里，于是：
    //   * 庄家只在第一小局算过一次 —— 之后每一小局的自风全是错的
    //     （用户原话：「场上各家对应的风还是只有第一小局能自动判断」）
    //   * 「失准」一旦立起来就再也没解除 —— 后面每小局的手牌都不再更新
    // 每小局发牌是一条**完整可信**的消息，本来就该拿它把状态全部归零。
    this.dealer = null;              // 本小局第一个打牌的人 = 庄家，重新认
    this._pendingDrawSeat = null;
    this.handSuspect = false;        // 新的一手是完整的，重新开始信
    this.desync = 0;
    this._touch();
    return { kind: 'newround', handNo: this.handNo,
             hand: this.hand.slice(), dora: this.dora.slice() };
  }

  _deal(d) {
    this.leftTileCount = d.leftTileCount || this.leftTileCount;
    if (d.dora && d.dora.length) this.dora = d.dora.slice();

    // **不能只看「有没有牌名」**。
    //
    // 原来我假设「带牌名的摸牌一定是我的」—— 错的。实测（录像是跑不变式）：
    // 座位 3 的摸牌也带着牌名发过来，于是被我当成自己的摸牌加进手牌，
    // 手牌就只增不减（13 -> 15 -> 16），之后整局全废。
    //
    // 正确的判据是**我自己的状态**：只有「手里正好 13 张等效、等着摸牌」时，
    // 这条摸牌才可能是我的。这也就是那条不变式本身（等效张数只能是 13 或 14）。
    // 判据分两层：
    //   主判据 = 座位（我摸的牌才带牌名，这条在真机上一直准）
    //   保险丝 = 等效张数（绝不能让它爬到 15/16 —— 那会让之后整局全废）
    //
    // 以前只看「有没有牌名」，于是别人的摸牌也进手牌，13->15->16。
    // 只看等效张数也不够：录像有断点时两条都不可靠。两层一起用最稳。
    if (this.handSuspect) { this._touch(); return { kind: 'draw', seat: d.seat, tile: d.tile || null, mine: false }; }
    // **带牌名的摸牌就是发给我的** —— 这是协议语义，不需要看座位，
    // 也就不受「手动换自风」影响（见 seatOverride 的注释）。
    const seatOk = true;
    // **「没满 14」就收，不是「正好 13」**。
    //
    // 我一开始要求正好 13，结果拔北被卡死：拔北先从手里扣掉那张北（13 -> 12），
    // 补摸的牌紧接着到（要在这个 12 上收）—— 正好 13 的条件把它拦掉了，
    // 于是每拔一次北就少一张牌，拔两次少两张，之后整局全废。
    //
    // 实测确认过：拔北之后同一座位**确实**会来一条带牌名的 DealTile（补摸）。
    // 所以判据放宽成「<= 13」—— 14 的时候还来摸牌才是真失控，那才拦。
    const roomOk = this.effective <= 13;
    const canDraw = d.tile && seatOk && roomOk;
    if (d.tile && seatOk && !roomOk) {
      // 座位对得上但手里已经满了 —— 说明前面漏了/多了动作（录像断点、重连）。
      // 说出来，并且**不入账**，免得一路错下去。
      this._markDesync(`座位对得上但手牌已满（等效 ${this.effective}），${d.tile} 没入账`);
    }
    if (canDraw) {
      // 座位投票：只有真正入账的那些才算数
      this._seatVotes[d.seat] = (this._seatVotes[d.seat] || 0) + 1;
      let best = null, bestN = 0;
      for (const [seat, n] of Object.entries(this._seatVotes)) {
        if (n > bestN) { bestN = n; best = Number(seat); }
      }
      if (best !== null && best !== this.mySeat) {
        const was = this.mySeat;
        this.mySeat = best;
        console.log(`[game] 座位推断：${SEAT_NAMES[best]}家（这一局 ${bestN} 票）` +
          (was === null ? '' : `　之前是 ${SEAT_NAMES[was]}家`));
      }
      this._pendingDrawSeat = d.seat;      // 摸牌的人下一手必须打牌（见 _discard）
      this.hand = sortHand(this.hand.concat([d.tile]));
      this.lastDraw = d.tile;
      this._touch();
      return { kind: 'draw', seat: d.seat, tile: d.tile, mine: true };
    }
    this._touch();
    return { kind: 'draw', seat: d.seat, tile: d.tile || null, mine: false };
  }


  _discard(d) {
    if (!d.tile) return null;
    // ---- 本局第一张打牌的人 = 庄家（麻将规则，不用推断）----
    //
    // 而且**庄家是 14 张起手、他第一个打牌** —— 这一手发生的时候
    // mySeat 很可能还是 null（还没摸过牌，没有「带牌名的摸牌」可依据）。
    //
    // 这就是「东家上来就失准」的根因：那一手没被认成我的 -> 手牌一直停在 14
    // -> 下一次摸牌被保险丝拦掉 -> 之后整局全废。
    //
    // 所以这里用规则直接定：第一张牌是谁打的谁就是庄家；
    // 而**我手里 14 张就说明我是庄家**（只有庄家起手 14 张）。
    if (this.dealer === null) {
      this.dealer = d.seat;
      if (this.mySeat === null && this.hand.length >= 14) {
        this.mySeat = d.seat;
        console.log(`[game] 座位定为 ${SEAT_NAMES[d.seat]}家（我是庄家：起手 ${this.hand.length} 张）`);
      }
    }

    // **座位校验**：刚才是谁摸的牌，这一手就该谁打。
    // 对不上就说明「带牌名的摸牌 = 我」这条推断错了 —— 直接纠正，
    // 而不是等手牌错开了再补救。
    if (this._pendingDrawSeat !== null && this._pendingDrawSeat !== d.seat) {
      console.log(`[game] 座位纠正：${SEAT_NAMES[this._pendingDrawSeat]}家刚摸牌，` +
        `却是 ${SEAT_NAMES[d.seat]}家打牌 —— 我应该是 ${SEAT_NAMES[this._pendingDrawSeat]}家`);
      this.mySeat = this._pendingDrawSeat;
    }
    if (this._pendingDrawSeat === d.seat) this._pendingDrawSeat = null;

    this.discards[d.seat].push(d.tile);
    const n = d.tile.charCodeAt(0) - 48;
    if (d.tile[1] === 'm' && n >= 2 && n <= 8) this.sanma = false;
    if (d.isLiqi) this.riichi[d.seat] = true;
    if (d.moqie) this.moqie[d.seat] += 1;

    // 一发：刚立直的那家进入一发；他自己下一次打牌就结束。
    // 别人副露也会打断 —— 那在 _meld 里清。
    this.ippatsu[d.seat] = false;          // 这家打牌了，他的一发（如果有）结束
    if (d.isLiqi) this.ippatsu[d.seat] = true;
    if (d.dora && d.dora.length) this.dora = d.dora.slice();

    const mine = this.mySeat !== null && d.seat === this.mySeat;
    if (mine) {
      const took = this._takeFromHand(d.tile);
      this.lastDraw = null;
      // **手牌和牌河对不上就说出来**。以前这里是静默的：打出去的牌不在手里时
      // 什么都不发生，手牌从此和真实牌局错开，界面照样显示，只是全错。
      // 实测踩过：屏幕上写着「12 张 + 1 副露（等效 15 张）」。
      if (!took) this._markDesync(`我打了 ${d.tile}，但手里没有这张`);
    }
    this._touch();
    return { kind: 'discard', seat: d.seat, tile: d.tile, mine,
             isLiqi: d.isLiqi, moqie: d.moqie };
  }

  _meld(d) {
    // 副露会打断所有人的一发
    this.ippatsu = [false, false, false, false];
    // 杠分三种，模型分得出「暗杠不算副露」和「加杠」：
    //   ActionChiPengGang type 2 = 大明槓（碰别人的牌再杠）
    //   ActionAnGangAddGang type 2 = 暗杠 / type 3 = 加杠（在 _kan 里）
    const type = d.type === 0 ? 'chi' : (d.type === 2 ? 'minkan' : 'pon');
    const meld = { type, tiles: (d.tiles || []).slice(), froms: (d.froms || []).slice() };
    this.meldsBySeat[d.seat].push(meld);          // 全家的都记
    const mine = this.mySeat !== null && d.seat === this.mySeat;
    // **保险丝**：副露只可能发生在「我手里 13 张等效、等着别人打牌」的时候。
    // 不是这个状态就说明前面已经错开了，这时候再往上加只会一路错下去
    //（实测：手牌涨到 16，之后整局全废）。所以拦住、记账、不打乱手牌。
    if (mine && (this.handSuspect || this.effective !== 13)) {
      if (!this.handSuspect) this._markDesync(`副露时手牌不是 13 张等效（当前 ${this.effective}）`);
      this._touch();
      return { kind: 'meld', seat: d.seat, meld, mine, skipped: true };
    }
    if (mine) {
      this.melds.push(meld);
      // 从手牌扣掉「自己出的那几张」。
      //
      // **不能靠 froms 判断哪张是别家的** —— 实测 froms 经常是空数组，
      // 那样就会一张都不扣（或者只扣一张），碰一次净 +1，手牌越打越多。
      //
      // 按规则扣最稳：碰/吃/杠都是「自己出 n-1 张 + 别人给 1 张」。
      const need = {};
      const mineCount = Math.max(0, meld.tiles.length - 1);
      // 别家给的那张 = 和副露者不同的那一张（froms 有的话用它更准）
      let gaveIdx = -1;
      if (meld.froms.length === meld.tiles.length) {
        gaveIdx = meld.froms.findIndex((f) => f !== d.seat);
      }
      const counted = {};
      for (let k = 0; k < meld.tiles.length; k++) {
        if (k === gaveIdx) continue;                 // 这张是别人打的
        const t = meld.tiles[k];
        counted[t] = (counted[t] || 0) + 1;
        if (counted[t] > mineCount) continue;        // 保险：最多扣 n-1 张
        need[t] = (need[t] || 0) + 1;
      }
      // 万一 froms 没给、又把别家的那张算进来了，这里按总数兜一下
      let total = Object.values(need).reduce((a, b) => a + b, 0);
      while (total > mineCount) {
        const t = Object.keys(need).find((k) => need[k] > 0);
        if (!t) break;
        need[t] -= 1; total -= 1;
      }
      for (const [t, c] of Object.entries(need)) this._takeFromHand(t, c);
      this.hand = sortHand(this.hand);

      // **事后校验**：扣完牌之后等效张数必须落在 13/14。
      //
      // 为什么不能只在前面拦：碰/吃要「从手牌扣掉 n-1 张」，而手牌一旦
      // 已经错开（前面漏了动作、中途重连），扣的动作就会少扣 ——
      // 表现是 13 直接跳到 15。前面那道闸门看不出来，只有扣完才知道。
      // 这时候**回滚**这一组副露，宁可少一组，也不要让手牌一路错下去
      //（曾经涨到 16，之后整局全废）。
      if (this.effective > 14) {
        this.melds.pop();
        this.hand = sortHand(this.hand.concat(
          Object.entries(need).flatMap(([t, c]) => Array(c).fill(t))));
        this._markDesync(`副露 ${type}(${(d.tiles || []).join('')}) 扣牌之后手牌变成了 ${this.effective} 张等效`);
        this._touch();
        return { kind: 'meld', seat: d.seat, meld, mine, rolledBack: true };
      }
    }
    this._touch();
    return { kind: 'meld', seat: d.seat, meld, mine };
  }

  _kita(d) {
    this.kita[d.seat] = (this.kita[d.seat] || 0) + 1;
    const mine = this.mySeat !== null && d.seat === this.mySeat;
    if (mine) {
      // 拔北只是从手里拿走一张北（并补摸一张），**不是副露** ——
      // 所以不进 this.melds，只记在 this.kita 里（见 effective 的注释）。
      const took = this._takeFromHand('4z');
      this.hand = sortHand(this.hand);
      // 扣不到北就说明手牌已经错开了：扣不掉会让手牌净 +1（补摸那张还在）。
      if (!took && !this.handSuspect) {
        this._markDesync(`拔北时手里没有北（手牌 ${this.hand.join(' ')}）`);
      }
    }
    if (d.dora && d.dora.length) this.dora = d.dora.slice();
    this._touch();
    return { kind: 'kita', seat: d.seat, mine };
  }

  _kan(d) {
    // type 2 = 暗杠（不破坏门清），type 3 = 加杠
    const kind = d.type === 3 ? 'kakan' : 'ankan';
    const myKan = { type: kind, tiles: d.tile ? [d.tile] : [], froms: [] };
    this.meldsBySeat[d.seat].push(myKan);
    const mine = this.mySeat !== null && d.seat === this.mySeat;
    if (mine && d.tile) {
      this.melds.push({ type: kind, tiles: [d.tile], froms: [] });
      this._takeFromHand(d.tile);      // 加杠：手里那张补进杠里
      this.hand = sortHand(this.hand);
    }
    this._touch();
    return { kind: 'kan', seat: d.seat, tile: d.tile, mine };
  }

  /** 给人看的一行摘要 */
  summary() {
    const parts = [];
    if (this.hand.length) parts.push(`手牌 ${this.hand.join(' ')}`);
    if (this.melds.length) {
      parts.push(this.melds.map((m) => `${m.type}(${m.tiles.join('')})`).join(' '));
    }
    if (this.dora.length) parts.push(`宝牌指示牌 ${this.dora.join(' ')}`);
    parts.push(this.discards.map((r, i) => `${SEAT_NAMES[i]}${r.length}`).join(' '));
    return parts.join('   |   ');
  }
}

module.exports = { GameState, SEAT_NAMES };
