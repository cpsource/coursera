import requests
import mwparserfromhell

title = "Dog"
url = f"https://en.wikipedia.org/w/index.php?title={title}&action=raw"
response = requests.get(url)
wikitext = response.text

wikicode = mwparserfromhell.parse(wikitext)
print(wikicode)

# Flexible, case-insensitive infobox filter
infoboxes = wikicode.filter_templates(matches=lambda t: t.name.lower().strip().startswith("infobox"))

if infoboxes:
    infobox = infoboxes[0]
    print("Extracted Scientific Classification:")
    for field in [
        "domain", "kingdom", "phylum", "classis", "ordo",
        "familia", "genus", "species"
    ]:
        if infobox.has(field):
            value = infobox.get(field).value.strip()
            print(f"{field.capitalize()}: {value}")
else:
    print("No infobox found.")

