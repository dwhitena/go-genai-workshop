package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	md "github.com/JohannesKaufmann/html-to-markdown"
)

// characterTextSplitter takes in a string and splits the string into
// chunks of a given size (split on whitespace) with an overlap of a
// given size of tokens (split on whitespace).
func characterTextSplitter(text string, splitSize int, overlapSize int) []string {

	// Create a slice to hold the chunks.
	chunks := []string{}

	// Split the text into tokens based on whitespace.
	tokens := strings.Split(text, " ")

	// Loop over the tokens creating chunks of size splitSize with an
	// overlap of overlapSize.
	for i := 0; i < len(tokens); i += splitSize - overlapSize {
		end := i + splitSize - overlapSize
		if end > len(tokens) {
			end = len(tokens)
		}
		chunks = append(chunks, strings.Join(tokens[i:end], " "))
	}
	return chunks
}

func main() {

	converter := md.NewConverter("", true, nil)

	// Download the Go contribution guide.
	res, err := http.Get("https://go.dev/doc/contribute")
	if err != nil {
		log.Fatal(err)
	}
	content, err := io.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		log.Fatal(err)
	}
	html := string(content)

	// Convert the html to markdown for convenience.
	markdown, err := converter.ConvertString(html)
	if err != nil {
		log.Fatal(err)
	}

	// Split the markdown string on "# Contribution Guide" to get the main bit.
	markdown = strings.Split(markdown, "# Contribution Guide")[1]

	// Split the text into reasonable size chunks with an overlap.
	chunks := characterTextSplitter(markdown, 100, 10)

	// Print out the first 3 chunks.
	fmt.Println(chunks[0])
	fmt.Println("")
	fmt.Println(len(chunks))
}
