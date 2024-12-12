// slightly complex example of Hermitian cancellation where more than 1 qubits are involved.

module {
  func.func @my_circuit(%in_qubit: !quantum.bit, %in_qubit_2: !quantum.bit, %angle: f64) -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
    %1:2 = quantum.custom "CNOT"() %0, %in_qubit_2 : !quantum.bit, !quantum.bit
    %2 = quantum.custom "Hadamard"() %1#0 : !quantum.bit
    %3 = quantum.custom "Hadamard"() %2 : !quantum.bit
    %4 = quantum.custom "RX"(%angle) %3 : !quantum.bit
    return %4, %1#1 : !quantum.bit, !quantum.bit
  }
}

// Expected output:
// module {
//   func.func @my_circuit(%in_qubit: !quantum.bit, %in_qubit_2: !quantum.bit, %angle: f64) -> (!quantum.bit, !quantum.bit) {
//     %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
//     %1:2 = quantum.custom "CNOT"() %0, %in_qubit_2 : !quantum.bit, !quantum.bit
//     %4 = quantum.custom "RX"(%angle) %1#0 : !quantum.bit
//     return %4, %1#1 : !quantum.bit, !quantum.bit
//   }
// }