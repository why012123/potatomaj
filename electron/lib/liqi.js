// 雀魂 ActionPrototype 的那层「加密」—— 解开它只用了 8 行。
//
// 算法来自 shinkuan/Akagi 的 src/bridge/majsoul/parser.rs（Apache-2.0）：
//
//     fn wtf_decode(data: &mut [u8]) {
//         const KEYS: [u8; 9] = [0x84,0x5E,0x4E,0x42,0x39,0xA2,0x1F,0x60,0x1C];
//         let base = 23 ^ data.len();
//         for (i, b) in data.iter_mut().enumerate() {
//             *b ^= (base + 5 * i + KEYS[i % 9]) as u8;
//         }
//     }
//
// 这不是密码学意义上的加密，是**位置相关的异或混淆**：密钥流由
// 「消息长度」和「字节位置」直接算出来，不需要任何会话密钥。
//
// 我之前花了很多轮去「破」它，还得出过「密钥流按消息长度分套、每种长度要单独
// 攒 10~15 局样本」的结论 —— 那个观察本身没错（base 确实依赖长度），
// 但结论错得很远：**它根本不需要样本，公式就在公开仓库里**。
//
// 这段注释留在代码里，是为了以后有人想「优化」它时先看清楚：
// 不要试图去 bit-twiddle，原样照抄 Akagi 的实现就是最稳的。

/** 9 字节循环密钥（Akagi 原样） */
const KEYS = [0x84, 0x5e, 0x4e, 0x42, 0x39, 0xa2, 0x1f, 0x60, 0x1c];

/**
 * 就地解混淆（返回新 Buffer，不动原数据）。
 * @param {Buffer|Uint8Array} buf ActionPrototype 的 data 字段
 */
function wtfDecode(buf) {
  const out = Buffer.from(buf);
  const base = 23 ^ out.length;
  for (let i = 0; i < out.length; i++) {
    out[i] ^= ((base + 5 * i + KEYS[i % KEYS.length]) & 0xff);
  }
  return out;
}

/** 反过来用是同一个函数（异或自反）—— 留着方便写测试 */
const wtfEncode = wtfDecode;

module.exports = { wtfDecode, wtfEncode, KEYS };
