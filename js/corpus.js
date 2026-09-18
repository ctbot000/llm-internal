/**
 * The single body of text this page learns from.
 *
 * Sections 6 and 7 both train on it, live, in the browser: the tokenizer merges
 * its character pairs, the generator counts its word sequences, and the training
 * demo runs gradient descent over its characters. It is original prose written
 * for this page, chosen to be plain, repetitive and short enough that a very
 * small model can find real structure in it.
 */

export const CORPUS = `
The model reads the sentence one token at a time. It reads the token, and it
reads the tokens that came before it, and from all of that it guesses the token
that comes next. There is nothing else in the machine. The guess is a number for
every word it knows, and the numbers are turned into chances, and one word is
drawn from those chances and written down.

A word on its own means very little. A word in a sentence means a great deal
more, because the words around it hold it in place. The model learns this by
reading a great deal of text and being wrong about the next word, over and over,
until it is wrong less often. Nobody tells it what a noun is. It is only ever
told that it guessed the wrong word, and by how much, and it moves a little in
the direction that would have been better.

The sea is grey in the morning and the sea is grey at night, and the light on
the water moves the way the words move in a sentence. A reader learns the shape
of a sentence long before the reader can say what the shape is. The model learns
the same shape from the same text, and it learns it as numbers, and the numbers
are the only thing it has.

What the model knows, it knows in its weights. What the model is thinking about
right now, it holds in its context. The weights do not change while it writes to
you. The context does, one token at a time, and when the context is full the
oldest part of it falls away and is gone.

It is a simple loop. Read the tokens, guess the next one, write it down, read
the tokens again. Run that loop a hundred times and you have a paragraph. Run it
on a large enough model, trained on a large enough amount of text, and the
paragraph is worth reading. The loop never changes. Only the guess gets better.

Everything else you hear about these models is a detail on top of that loop.
Some of the details matter a great deal. None of them replace the loop.
`.trim().replace(/\s*\n\s*/g, ' ');

/** Lower-cased, reduced to a tiny alphabet, for the character-level trainer. */
export const CHAR_TEXT = CORPUS
  .toLowerCase()
  .replace(/[^a-z ,.]/g, ' ')
  .replace(/ +/g, ' ')
  .trim();

/**
 * Frequent English words, most common first. The tokenizer trains on the corpus
 * plus this list (weighted by rank) so it behaves like a real one: the words a
 * model sees constantly survive as single tokens, and everything else breaks up.
 */
export const COMMON_WORDS = `the be to of and a in that have I it for not on with he as you do at this
but his by from they we say her she or an will my one all would there their what so up out if about who get which go me
when make can like time no just him know take people into year your good some could them see other than then now look
only come its over think also back after use two how our work first well way even new want because any these give day
most us is are was were been being has had did does said made made find tell ask seem feel try leave call great little
own old right big high different small large next early young important few public bad same able man woman child world
school state family student group country problem hand part place case week company system program question work
number night point home water room mother area money story fact month lot book eye job word business issue side kind
head house service friend father power hour game line end member law car city community name president team minute
idea kid body information back parent face others level office door health person art war history party result change
morning reason research girl guy moment air teacher force education foot boy age policy process music market sense
nation plan college interest death experience effect use class control care field development role effort rate heart
drug show leader light voice wife police mind price report decision son view relationship town road arm difference
value building action model season society tax director position player record paper space ground form event official
matter center couple site project activity star table need court produce learn read write speak listen build train
token word sentence text number guess chance loop weight context detail language machine`
  .trim().split(/\s+/);
