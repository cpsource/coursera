import mwparserfromhell
import requests

wikitext = requests.get(
    "https://en.wikipedia.org/w/index.php?title=Dog&action=raw"
).text

wikicode = mwparserfromhell.parse(wikitext)
infobox = wikicode.filter_templates(matches="Infobox")[0]
print(infobox.name, infobox.params)
