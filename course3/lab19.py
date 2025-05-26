import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='MyApp/1.0 (page.cal@gmail.com)'  # use a real email if possible
)
page = wiki.page("Dog")

print(page.summary)
print(page.sections)

