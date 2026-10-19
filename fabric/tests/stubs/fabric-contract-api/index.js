// test-only stand-in for fabric-contract-api (so tests run without installing Fabric packages)
class Contract { constructor(name) { this.__name = name; } }
module.exports = { Contract };
