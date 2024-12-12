// Simple circuit with Hadamard conjugation

module {
  func.func @my_circuit(%in_qubit: !quantum.bit) -> !quantum.bit {
    %0 = quantum.custom "PauliX"() %in_qubit : !quantum.bit
    %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
    %2 = quantum.custom "PauliX"() %1 : !quantum.bit
    %3 = quantum.custom "Hadamard"() %2 : !quantum.bit
    return %3 : !quantum.bit
  }
}

// Expected Output
//module {
//  func.func @my_circuit(%arg0: !quantum.bit) -> !quantum.bit {
//    %0 = quantum.custom "PauliX"() %arg0 : !quantum.bit
//    %1 = quantum.custom "PauliZ"() %0 : !quantum.bit
//    return %1 : !quantum.bit
//  }
//}
