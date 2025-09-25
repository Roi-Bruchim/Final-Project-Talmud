import scrapy
from scrapy.crawler import CrawlerProcess
import re


def parse_word(response):
    word = response.meta['word']

    # Extracting lexicons
    word['lexicon_0'] = response.xpath('//body/br[2]/following-sibling::text()').get()
    if word['lexicon_0'] and word['lexicon_0'].strip() == '':
        word['lexicon_0'] = response.xpath('//body/br[3]/following-sibling::text()').get()

    for i in range(1, 5):
        lex = response.xpath(f'//body/hr[{i}]/following-sibling::text()').get()
        if lex and lex.strip() != '':
            word[f'lexicon_{i}'] = lex.strip()
        else:
            word[f'lexicon_{i}'] = None

    # Extract split words from <span class="lem"><font>
    split_words = response.xpath('//span[@class="lem"]/font/text()').getall()
    split_words = [w.strip() for w in split_words if w.strip()]  # Clean up whitespace and empty strings

    for i in range(4):
        word[f'split_word_{i}'] = split_words[i] if i < len(split_words) else None

    # Check if the word is a verb (look for "verb" in lexicon_0, lexicon_1, lexicon_2, lexicon_3)
    verb_form = None
    for lexicon_key in ['lexicon_0', 'lexicon_1', 'lexicon_2', 'lexicon_3']:
        if word.get(lexicon_key) and 'verb' in word[lexicon_key]:
            verb_form = word[lexicon_key].strip().split(' ')[2]  # Extract the form after 'verb'
            break

    # Extract meanings from <span class="mg1">
    if verb_form:
        translation_list = response.xpath(
            f'(//span[@class="bin" and text()="{verb_form}"]/following-sibling::span[@class="mg1"])[position()>1]/text()'
        ).getall()

        # Clean up translations: filter out numbers and empty strings
        cleaned_translations = []
        for t in translation_list:
            t = t.strip()
            if t and not re.match(r'^\d+$', t):  # Exclude pure numbers
                cleaned_translations.append(t)

        for i in range(4):
            word[f'meaning_{i}'] = cleaned_translations[i] if i < len(cleaned_translations) else None

    else:
        meanings = response.xpath('//span[@class="mgP"]/text()').getall()
        meanings = [m.strip() for m in meanings if m.strip()]  # Clean up whitespace and empty strings

        for i in range(4):
            word[f'meaning_{i}'] = meanings[i] if i < len(meanings) else None

    yield word

class CalWordSpider(scrapy.Spider):
    name = 'cal'
    start_urls = [
        'https://cal.huc.edu/get_a_chapter.php?file=71037'
    ]

    def parse(self, response):
        for el in response.css('tr > td:nth-child(2) > a'):
            word = {
                'text': el.css('::text').get(),
                'url': el.xpath('@href').get()
            }
            yield scrapy.Request(
                url=f"https://cal.huc.edu/{word['url']}",
                meta={'word': word},
                callback=parse_word
            )


# Run the spider
process = CrawlerProcess(settings={
    'FEEDS': {
        'Masekhet_split_37.json': {
            'format': 'json',
            'overwrite': True
        }
    }
})

process.crawl(CalWordSpider)
process.start()