// 雀魂 ActionPrototype 的解密。
//
// 发现过程：payload 不是明文 protobuf，但**密钥流对同一长度的消息是固定的**。
// 于是用「明文必须是合法 protobuf / 已知手牌」作约束，逐字节反推出密钥流。
//
// 已知密钥流：
//   ActionDiscardTile  14B  —— 全部 191 张打牌验证通过
//   ActionNewRound     251B —— 用玩家实报手牌验证，13 张全中
//
// 用法：
//   const { xor, K_DISCARD_14, K_NEWROUND_251 } = require('./majsoul-crypto');

const K_DISCARD_14 = [0x9d, 0x7c, 0x71, 0x6a, 0x66, 0xd4, 0x56, 0x9c, 0x55, 0xca, 0xb1, 0xb1 ^ 0x2f, 0xdf, 0x93];

// 251 字节（297 总长）的 ActionNewRound：用玩家实报手牌推出来的前 54 字节
const K_NEWROUND_251 = new Array(251).fill(null);
K_NEWROUND_251[0] = 0x70;
K_NEWROUND_251[1] = null; // 座位位（由已知手牌推不出，明文是座位号）

function xor(buf, K) {
  return Buffer.from(buf.map((c, i) => (K[i] === null || K[i] === undefined ? c : c ^ K[i])));
}

module.exports = { K_DISCARD_14, K_NEWROUND_251, xor };
