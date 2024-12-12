// Simple example of Hermitian cancellation where two adjacent Hadamard gates are removed.

module {
  func.func @my_circuit(%in_qubit: !quantum.bit, %angle: f64) -> !quantum.bit {
    %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
    %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %3 = quantum.custom "RX"(%angle) %2 : !quantum.bit
    return %3 : !quantum.bit
  }
}

// Expected output:
// module {
//   func.func @my_circuit(%in_qubit: !quantum.bit, %angle: f64) -> !quantum.bit {
//     %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
//     %1 = quantum.custom "RX"(%angle) %0 : !quantum.bit
//     return %1 : !quantum.bit
//   }
// }
