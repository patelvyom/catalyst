// slightly complex test case where propagation continues in chain:
// X_1 CNOT_12 CNOT_12 -> CNOT_12 X_1 X_2 CNOT_12 -> CNOT_12 CNOT_12 X_1 X_2 X_2

module {
  func.func @my_circuit() -> (!quantum.bit, !quantum.bit) {
    %0 = quantum.alloc( 2) : !quantum.reg
    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
    %3 = quantum.custom "PauliX"() %1 : !quantum.bit
    %out_qubits:2 = quantum.custom "CNOT"() %3, %2 : !quantum.bit, !quantum.bit
    %out_qubits_1:2 = quantum.custom "CNOT"() %out_qubits#0, %out_qubits#1 !quantum.bit, !quantum.bit
    return %out_qubits_1#0, %out_qubits_1#1 : !quantum.bit, !quantum.bit
  }
}

// Expected Output
//module {
//  func.func @my_circuit( ) -> (!quantum.bit, !quantum.bit) {
//    %0 = quantum.alloc( 2) : !quantum.reg
//    %1 = quantum.extract %0[ 0] : !quantum.reg -> !quantum.bit
//    %2 = quantum.extract %0[ 1] : !quantum.reg -> !quantum.bit
//    %out_qubits:2 = quantum.custom "CNOT"() %1, %2 : !quantum.bit, !quantum.bit
//    %out_qubits_1:2 = quantum.custom "CNOT"() %out_qubits#0, %out_qubits#1 : !quantum.bit, !quantum.bit
//    %3 = quantum.custom "PauliX"() %out_qubits_1#0 : !quantum.bit
//    %4 = quantum.custom "PauliX"() %out_qubits_1#1 : !quantum.bit
//    %5 = quantum.custom "PauliX"() %4 : !quantum.bit
//    return %3, %5 : !quantum.bit, !quantum.bit
//  }
//}

