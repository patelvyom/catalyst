// Hermitian cancellation and cnot cancellation where cnot cancellation is nested between already 
// nested Hermitian ops.

module {
  func.func @my_circuit(%in_qubit: !quantum.bit, %in_qubit_2: !quantum.bit, %angle: f64) -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
    %1:2 = quantum.custom "CNOT"() %0, %in_qubit_2 : !quantum.bit, !quantum.bit
    %2 = quantum.custom "PauliX"() %1#0 : !quantum.bit
    %3:2 = quantum.custom "CNOT"() %2, %1#1 : !quantum.bit, !quantum.bit
    %4 = quantum.custom "Hadamard"() %3#0 : !quantum.bit
    %5 = quantum.custom "Hadamard"() %4 : !quantum.bit
    %6:2 = quantum.custom "CNOT"() %5, %3#1 : !quantum.bit, !quantum.bit
    %7 = quantum.custom "PauliX"() %6#0 : !quantum.bit
    return %7, %6#1 : !quantum.bit, !quantum.bit
  }
}

// Expected output:
// module {
//   func.func @my_circuit(%in_qubit: !quantum.bit, %in_qubit_2: !quantum.bit, %angle: f64) -> (!quantum.bit, !quantum.bit) {
//     %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
//     %1:2 = quantum.custom "CNOT"() %0, %in_qubit_2 : !quantum.bit, !quantum.bit
//     return %1#0, %1#1 : !quantum.bit, !quantum.bit
//   }
// }