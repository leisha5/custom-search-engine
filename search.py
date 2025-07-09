import os
import math
import re
import pytest

def clean(token: str, pattern: re.Pattern[str] = re.compile(r"\W+")) -> str: 
    """Returns all the characters in the token lowercased and without matches to the given pattern.
    >>> clean("Hello!")'hello'"""
    return pattern.sub("", token.lower())


class Document:
    def __init__(self, filename: str) -> None:
        """ 
        Takes in a string path.
        Initialises a dictionary with each unique word mapped to its float frequency,
        calculated by the number of times a word appears divided by the total number of words.
        Not case-specific, ignoring whitespace, and any special characters. Returns None.
        """
        self.filename: str = filename
        frequency = {}
        with open(filename) as file:
            tokens = file.read().split()
        for word in tokens:
            word = clean(word)
            if word not in frequency: # using float(0) because otherwise mypy throws a fit
                frequency[word] = float(0) # cannot assign a float to a key␣ ↪that originally mapped to an integer
                frequency[word] += float(1) # if you had an else statement, the if␣ ↪statement would be frequency[word]= float(1)
        for unique_word in list(frequency.keys()):
            frequency[unique_word] = frequency[unique_word] / len(tokens)
            # I interpreted the total number of words as the total number of tokens
            # as opposed to the total number of cleaned/normalised words
        self.frequency = frequency
        
    def term_frequency(self, term: str) -> float:
        """
        Takes in a given str term.
        Returns its float frequency.
        If term does not exist in the precomputed dictionary, returns 0.
        """
        term = clean(term)
        if term in self.frequency.keys():
            return self.frequency[term]
        else:
            return float(0)
        # can get rid of if/else by using .get()?

    def get_path(self) -> str:
        """
        Returns the string path of the file.
        """
        return self.filename
    
    def get_words(self) -> set[str]:
        """
        Returns a set of the unique words in the document.
        Not case-specific, ignoring whitespace, and any special characters.
        """
        return set(self.frequency.keys())
    
    def __repr__(self) -> str:
        """
        Returns the string representation of the document in the format Document('{path}').
        """
        return "Document('{" + self.filename + "}')"
    

class TestDocument:
    euro = Document("small_wiki/Euro - Wikipedia.html")
    doc1 = Document("doggos/doc1.txt")

    def test_term_frequency(self) -> None:
        assert self.euro.term_frequency("Euro") == pytest.approx(0.0086340569495348)
        assert self.doc1.term_frequency("dogs") == pytest.approx(1 / 5) 
        
    def test_get_words(self) -> None:
        assert set(w for w in self.euro.get_words() if len(w) == 1) == set([*"0123456789acefghijklmnopqrstuvxyz".lower()]) # All one-letter words in Euro
        assert self.doc1.get_words() == set("dogs are the greatest pets".split())
    
    def test_get_path(self) -> None:
        assert self.euro.get_path() == "small_wiki/Euro - Wikipedia.html"
        assert self.doc1.get_path() == "doggos/doc1.txt"
        
    def test_repr(self) -> None:
        assert self.euro.__repr__() == "Document('{small_wiki/Euro - Wikipedia.html}')"
        assert self.doc1.__repr__() == "Document('{doggos/doc1.txt}')" 

path = "doggos"
extension = ".txt"
for filename in os.listdir(path):
    if filename.endswith(extension): print(os.path.join(path, filename))

class SearchEngine:
    def __init__(self, path: str, extension: str =".txt"):
        """
        Takes in a string path to a directory, and a string file extension. Default file extension is .txt
        Constructs an inverted index from the files in the specified directory matching the given extension,
        this is a dictionary that maps each unique word in all of the files to which files they appear in,
        in the form {"word": [file1, file 2], ...}.
        Assumes the string represents a valid directory, and that the directory only contains valid files.
        """
        files = set()
        inverted_index: dict[str, list[Document]] = {}
        for filename in os.listdir(path):
            if filename.endswith(extension):
                file = Document(os.path.join(path, filename)) # gives all the files in the directory as Document objects files.add(file)
                for word in file.get_words():
                    if word not in inverted_index.keys():
                        inverted_index[word] = []
                    inverted_index[word].append(file)
        self.inverted_index = inverted_index
        self.files = files
        self.path = path

    def _calculate_idf(self, term: str) -> float:
        """
        Takes in a string term.
        Returns the inverse document frequency of the term.
        If the term is not in the corpus, returns 0.
        """
        if term in self.inverted_index:
            return math.log(len(self.files) / len(self.inverted_index[term]))
        else:
            return 0
        
    def search(self, query: str) -> list[str]:
        """
        Takes in a string query consisting of one or more terms.
        Returns a list of relevant document paths that match at least one of
        the cleaned terms (sorted by descending tf-idf statistic).
        If there are no matching documents, returns an empty list.
        """
        words = query.split()
        output = {}
        for word in words:
            term = clean(word)
            if term in self.inverted_index:
                for file in self.inverted_index[term]:
                    tf_idf = file.term_frequency(term) * self._calculate_idf(term)
                    if file not in output:
                        output[file.get_path()] = float(0)
                    output[file.get_path()] = tf_idf
                    # using .get_path() here because otherwise it maps to the full path (including the directory)
                    # instead of just the filename, which is what I want
        final_output = sorted(output.keys(), key=lambda x: output[x], reverse=True)
        return final_output

    def __repr__(self) -> str:
        """
        Returns a string representation of this search engine, in the format␣ ↪SearchEngine('{path}').
        """
        return "SearchEngine('{" + self.path + "}')"

class TestSearchEngine:
    doggos = SearchEngine("doggos")
    small_wiki = SearchEngine("small_wiki", ".html")
    testing_search = SearchEngine("testing_search")
    def test_calculate_idf(self) -> None:
        assert self.doggos._calculate_idf("bird") == 0
        assert self.small_wiki._calculate_idf("seattle") == pytest.approx(1.94591)
        assert self.testing_search._calculate_idf("whom") == pytest.approx(math.log(2))

    def test__repr__(self) -> None:
       assert self.doggos.__repr__() == "SearchEngine('{doggos}')"
       assert self.small_wiki.__repr__() == "SearchEngine('{small_wiki}')"
       assert self.testing_search.__repr__() == "SearchEngine('{testing_search}')"

    def test_search(self) -> None:
        assert self.doggos.search("love dogs") == ["doggos/doc3.txt", "doggos/doc1.txt"]
        assert self.small_wiki.search("data")[:10] == [
           "small_wiki/Internet privacy - Wikipedia.html",
           "small_wiki/Machine learning - Wikipedia.html",
           "small_wiki/Bloomberg L.P. - Wikipedia.html",
           "small_wiki/Waze - Wikipedia.html",
           "small_wiki/Digital object identifier - Wikipedia.html",
           "small_wiki/Chief financial officer - Wikipedia.html",
           "small_wiki/UNCF - Wikipedia.html",
           "small_wiki/Jackson 5 Christmas Album - Wikipedia.html",
           "small_wiki/KING-FM - Wikipedia.html",
           "small_wiki/The News-Times - Wikipedia.html",
            ]
        assert self.testing_search.search("exclamation mark") == ["testing_search/searchtestfile2.txt","testing_search/searchtestfile1.txt"]

SearchEngine("small_wiki", ".html").search("data")