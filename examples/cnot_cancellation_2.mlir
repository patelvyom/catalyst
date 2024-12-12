// A slightly more compilated example with arrangement: CNOT_12 , CNOT_23, CNOT_12, CNOT_12 where
// we are testing that CNOT_12 and CNOT_23 don't get cancelled out.

module {
  func.func @my_circuit() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 4) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
    %4 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit 
    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
    %out_qubits_0:2 = quantum.custom "CNOT"() %2, %3 : !quantum.bit, !quantum.bit
    %out_qubits_1:2 = quantum.custom "CNOT"() %out_qubits#0, %out_qubits_0#0 : !quantum.bit, !quantum.bit
    %out_qubits_2:2 = quantum.custom "CNOT"() %out_qubits_1#0, %out_qubits_1#1 : !quantum.bit, !quantum.bit
    return %out_qubits_2#0, %out_qubits_2#1 : !quantum.bit, !quantum.bit
  }
}

// Expected Output
// module {
//   func.func @my_circuit() -> (!quantum.bit, !quantum.bit) {
//     %0 = quantum.alloc( 4) : !quantum.reg
//     %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
//     %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
//     %3 = quantum.extract %0[ 2] : !quantum.reg -> !quantum.bit
//     %4 = quantum.extract %0[ 3] : !quantum.reg -> !quantum.bit 
//     %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
//     %out_qubits_0:2 = quantum.custom "CNOT"() %2, %3 : !quantum.bit, !quantum.bit
//     return %out_qubits_0#0, %out_qubits_0#1 : !quantum.bit, !quantum.bit
//   }
// }
