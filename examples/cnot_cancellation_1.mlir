// Description: A simple example of a quantum circuit with two adjacent CNOT gates.

module {
  func.func @my_circuit() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
    return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
  }
}

// Expected Output
// module {
//   func.func @my_circuit() -> (!quantum.bit, !quantum.bit) {
//     %0 = quantum.alloc( 2) : !quantum.reg
//     %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
//     %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
//     return %1, %2 : !quantum.bit, !quantum.bit
//   }
// }
