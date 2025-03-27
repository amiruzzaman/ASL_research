// Polyfill to Symbol.dispose

if (!Symbol.dispose) {
  Object.defineProperty(Symbol, "dispose", {
    value: Symbol("dispose"),
    writable: false,
    enumerable: false,
    configurable: false,
  });
}
