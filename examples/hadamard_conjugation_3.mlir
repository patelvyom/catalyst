// Complex example of Hadamard conjugation. Here we have nested Hadamard conjugations. 
// H (HXH) H --> HXH -> Z 

module {
  func.func @my_circuit(%in_qubit: !quantum.bit, %angle: f64) -> !quantum.bit {
    %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
    %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
    %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
    %3 = quantum.custom "RX"(%angle) %2 : !quantum.bit
    %4 = quantum.custom "Hadamard"() %3 : !quantum.bit

    %5 = quantum.custom "Hadamard"() %4 : !quantum.bit
    %6 = quantum.custom "PauliX"() %5 : !quantum.bit
    %7 = quantum.custom "Hadamard"() %6 : !quantum.bit

    %8 = quantum.custom "Hadamard"() %7 : !quantum.bit
    %9 = quantum.custom "Hadamard"() %8 : !quantum.bit
    %10 = quantum.custom "PauliZ"() %9 : !quantum.bit
    %11 = quantum.custom "Hadamard"() %10 : !quantum.bit
    return %11 : !quantum.bit
  }
}

// Expected output:
// module {
//   func.func @my_circuit(%in_qubit: !quantum.bit, %angle: f64) -> !quantum.bit {
//     %0 = quantum.custom "RX"(%angle) %in_qubit : !quantum.bit
//     %1 = quantum.custom "Hadamard"() %0 : !quantum.bit
//     %2 = quantum.custom "Hadamard"() %1 : !quantum.bit
//     %3 = quantum.custom "RX"(%angle) %2 : !quantum.bit
//
//     %4 = quantum.custom "PauliX"() %3 : !quantum.bit
//     %5 = quantum.custom "PauliX"() %4 : !quantum.bit
//     return %5 : !quantum.bit
//   }
// }
