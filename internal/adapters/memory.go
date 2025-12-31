package adapters

/*
#cgo CFLAGS: -I${SRCDIR}/../../external/llama.cpp/include
#cgo LDFLAGS: -L${SRCDIR}/../../build/bin -lllama -Wl,-rpath,$ORIGIN/build/bin

#include <stdlib.h>
#include <stdio.h>
#include <llama.h>
*/
import "C"
import "log"

// Clear the memory contents
// If data == true, the data buffers will also be cleared together with the metadata
func ClearMemory(mem MemoryT, data bool) {
	C.llama_memory_clear(mem, C.bool(data))
}

// Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
// Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
// seq_id < 0 : match any sequence
// p0 < 0     : [0,  p1]
// p1 < 0     : [p0, inf)
func RemoveMemorySequence(mem MemoryT, seqId int32, p0 int32, p1 int32) {
	if !C.llama_memory_seq_rm(mem, C.int32_t(seqId), C.int32_t(p0), C.int32_t(p1)) {
		log.Fatalf("RemoveMemorySequence: position %d to %d in sequence %d cannot be removed\n", p0, p1, seqId)
	}
}

// Copy all tokens that belong to the specified sequence to another sequence
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
func CopyMemorySequence(mem MemoryT, seqIdSrc int32, seqIdDst int32, p0 int32, p1 int32) {
	C.llama_memory_seq_cp(mem, C.int32_t(seqIdSrc), C.int32_t(seqIdDst), C.int32_t(p0), C.int32_t(p1))
}

// Removes all tokens that do not belong to the specified sequence
func KeepMemorySequence(mem MemoryT, seqId int32) {
	C.llama_memory_seq_keep(mem, C.int32_t(seqId))
}

// Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
func AddMemorySequence(mem MemoryT, seqId int32, p0 int32, p1 int32, delta int32) {
	C.llama_memory_seq_add(mem, C.int32_t(seqId), C.int32_t(p0), C.int32_t(p1), C.int32_t(delta))
}

// Integer division of the positions by factor of `d > 1`
// p0 < 0 : [0,  p1]
// p1 < 0 : [p0, inf)
func DivideMemorySequence(mem MemoryT, seqId int32, p0 int32, p1 int32, d int) {
	C.llama_memory_seq_div(mem, C.int32_t(seqId), C.int32_t(p0), C.int32_t(p1), C.int(d))
}

// Returns the smallest position present in the memory for the specified sequence
// This is typically non-zero only for SWA caches
// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
// Return -1 if the sequence is empty
func MinMemorySequence(mem MemoryT, seqId int32) int32 {
	return int32(C.llama_memory_seq_pos_min(mem, C.int32_t(seqId)))
}

// Returns the largest position present in the memory for the specified sequence
// Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
// Return -1 if the sequence is empty
func MemorySequenceMax(mem MemoryT, seqId int32) int32 {
	return int32(C.llama_memory_seq_pos_max(mem, C.int32_t(seqId)))
}

// Check if the memory supports shifting
func MemoryCanShift(mem MemoryT) bool {
	return bool(C.llama_memory_can_shift(mem))
}
