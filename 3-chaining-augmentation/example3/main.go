package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"math"
	"os"

	cohere "github.com/cohere-ai/cohere-go"
)

// embed vectorizes a user message/query.
func embed(message string, co *cohere.Client) ([]float64, error) {
	res, err := co.Embed(cohere.EmbedOptions{
		Model: "embed-english-light-v2.0",
		Texts: []string{message},
	})
	if err != nil {
		return nil, err
	}
	return res.Embeddings[0], nil
}

// VectorizedChunk is a struct that holds a vectorized chunk.
type VectorizedChunk struct {
	Chunk  string    `json:"chunk"`
	Vector []float64 `json:"vector"`
}

// VectorizedChunks is a slice of vectorized chunks.
type VectorizedChunks []VectorizedChunk

// cosineSimilarity calculates the cosine similarity between two vectors.
func cosineSimilarity(a []float64, b []float64) (cosine float64, err error) {
	count := 0
	length_a := len(a)
	length_b := len(b)
	if length_a > length_b {
		count = length_a
	} else {
		count = length_b
	}
	sumA := 0.0
	s1 := 0.0
	s2 := 0.0
	for k := 0; k < count; k++ {
		if k >= length_a {
			s2 += math.Pow(b[k], 2)
			continue
		}
		if k >= length_b {
			s1 += math.Pow(a[k], 2)
			continue
		}
		sumA += a[k] * b[k]
		s1 += math.Pow(a[k], 2)
		s2 += math.Pow(b[k], 2)
	}
	if s1 == 0 || s2 == 0 {
		return 0.0, errors.New("vectors should not be null (all zeros)")
	}
	return sumA / (math.Sqrt(s1) * math.Sqrt(s2)), nil
}

// search through the vectorized chunks to find the most similar chunk.
func search(chunks VectorizedChunks, embedding []float64) (string, error) {
	outChunk := ""
	var maxSimilarity float64 = 0.0
	for _, c := range chunks {
		distance, err := cosineSimilarity(c.Vector, embedding)
		if err != nil {
			return "", err
		}
		if distance > maxSimilarity {
			outChunk = c.Chunk
			maxSimilarity = distance
		}
	}
	return outChunk, nil
}

func main() {

	// Connect to Cohere.
	apiKey := os.Getenv("COHERE_API_KEY")
	if apiKey == "" {
		fmt.Fprintln(os.Stderr, "COHERE_API_KEY not specified")
		os.Exit(1)
	}
	co, err := cohere.CreateClient(apiKey)
	if err != nil {
		log.Fatal(err)
	}

	// Open the JSON file and load in the vectorized embeddings.
	f, err := os.Open("../example2/chunks.json")
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	// Load it into the chunks value.
	var chunks VectorizedChunks
	if err := json.NewDecoder(f).Decode(&chunks); err != nil {
		log.Fatal(err)
	}

	// Embed a question for the RAG answer.
	message := "What is the review process?"
	embedding, err := embed(message, co)
	if err != nil {
		log.Fatal(err)
	}

	// Search for the relevant chunk.
	chunk, err := search(chunks, embedding)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(chunk)
}
