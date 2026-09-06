import { pipeline, env } from '@huggingface/transformers';

// Configure transformers.js to load 100% locally from /models/Xenova (Zero Remote Downloads / Offline)
env.allowLocalModels = true;
env.allowRemoteModels = false;
env.useBrowserCache = false;

// Purge any stale/corrupted cache from previous failed browser loads
if (typeof window !== 'undefined' && 'caches' in window) {
  caches.delete('transformers-cache').catch(() => {});
}

// Lines that only contain structured key-value fields already 100% matched by deterministic regex
const SKIP_PATTERNS = /^\s*(?:\*|-|\d+\.)?\s*(?:Date of Birth|Mobile|Landline|Alternate|Aadhaar|Masked|PAN|Voter|Driving|Vehicle|GSTIN|Bank IFSC|Bank Account|Account Number|Routing Transit|SWIFT|UPI|SSN|Belgian|Amex|Eurozone|Credit Card|Card Verification|Corporate Mobile|Policy Group|Individual Member|Employee Identification|Freight Forwarder|Airline Ticket|Digital Fingerprint|Secure PGP|Corporate IP|Direct Line|Gender Identity|Marital Status|Age|Blood Type|Expiration Date|TAN|CIN|ITIN|NINO|ABHA|UAN)\b/i;

let cachedNERPipeline = null;
export const activeNEREngineName = 'Local Xenova BERT-NER (Offline WASM) + Regex';

export async function getNERPipeline() {
  if (!cachedNERPipeline) {
    cachedNERPipeline = await pipeline('token-classification', '/models/Xenova', {
      quantized: true,
      subfolder: '',
      local_files_only: true
    });
  }
  return { ner: cachedNERPipeline, engine: activeNEREngineName };
}

// Comprehensive UI, E-Commerce, Navigation and Button Stopwords (Never PII)
export const UI_STOPWORDS = new Set([
  'home', 'account', 'my account', 'order', 'orders', 'my orders', 'my', 'invoice',
  'download', 'download invoice', 'return', 'exchange', 'cart', 'more', 'seller',
  'become a seller', 'item', 'items', 'search', 'search for products', 'products',
  'brands', 'view', 'details', 'price', 'price details', 'total', 'total amount',
  'discount', 'other discount', 'delivery', 'delivery details', 'delivered', 'confirmed',
  'order confirmed', 'updates', 'see all updates', 'policy', 'return policy', 'valid',
  'know', 'know more', 'chat', 'chat with us', 'offers', 'offers earned', 'gift',
  'rate', 'experience', 'rate your experience', 'reward', 'extra', 'extra cost', 'free',
  'special', 'special price', 'listing', 'listing price', 'amount', 'fees', 'total fees',
  'netf', 'lix', 'netflix', 'flipkart', 'cultx', 'shoes', 'men', 'women', 'kids',
  'baby & kids', 'electronics', 'appliances', 'tvs & appliances', 'home & furniture',
  'furniture', 'sports', 'sports, books & more', 'flights', 'offer zone', 'zone',
  'save', 'cancel', 'submit', 'select', 'try', 'sample', 'icon', 'logo', 'button',
  'close', 'open', 'menu', 'nav', 'header', 'footer', 'next', 'prev', 'back', 'help',
  'login', 'logout', 'sign in', 'sign out', 'settings', 'profile'
]);

// Company, Organization, Corporate & Brand detection filter (never PII)
export const COMPANY_INDICATORS = /\b(?:Inc|Corp|Corporation|Ltd|Limited|LLC|LLP|Pvt|Private|GmbH|AG|SA|BV|NV|Bank|Banque|Labs|Laboratories|Technologies|Technology|Tech|Enterprises|Solutions|Services|Ventures|Holdings|Group|Co|Company|International|Global|Center|Hospital|Clinic|Pharma|Biopharma|University|College|Institute|Store|Seller|Shop|Retail|Studio|Agency|Brand|Jeans|Denim|Clothing|Fashion|Apparel|Wear|Outfitters|Garments|Pepe|Zara|Nike|Adidas|Puma|Levis?)\b/i;

// Valid ISO 3166-1 alpha-2 country codes used in real international IBANs
const IBAN_COUNTRIES = 'AL|AD|AT|AZ|BH|BE|BA|BR|BG|CR|HR|CY|CZ|DK|DO|EE|FO|FI|FR|GE|DE|GI|GR|GL|GT|HU|IS|IE|IL|IT|JO|KZ|XK|KW|LV|LB|LI|LT|LU|MK|MT|MR|MU|MD|MC|ME|NL|NO|PK|PS|PL|PT|QA|RO|SM|SA|RS|SK|SI|ES|SE|CH|TN|TR|AE|GB|VA';

// --- 1. COMPREHENSIVE REGEX SUITE (INDIAN + GLOBAL) ---
export const REGEX_RULES = [
  // 1. Email Addresses
  { pattern: /\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b/gi, tag: 'EMAIL' },

  // 2. UPI IDs / VPA
  { pattern: /\b[a-zA-Z0-9.\-_]{2,64}@(okaxis|okhdfcbank|okicici|oksbi|paytm|ybl|ibl|upi|axl|apl|barodampay|postbank|kotak|icici|sbi|hdfcbank)\b/gi, tag: 'UPI_ID' },

  // 3. Social Media & Platform Handles (@username, @ rohitsinghal)
  { pattern: /(?<=\s|^|[([:;,])@\s*[a-zA-Z0-9_.-]{2,32}\b/gi, tag: 'HANDLE' },

  // 4. Labeled Handles & User IDs (ID: #ROHITSINGHAL, User ID: abc_12)
  { pattern: /(?:Handle|Username|User\s*ID|ID)\s*[:#]\s*#?\s*([a-zA-Z0-9_.-]{3,32})\b/gi, tag: 'HANDLE' },

  // 5. URLs & Personal Websites
  { pattern: /\b(?:https?:\/\/|www\.)[-a-zA-Z0-9@:%._+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:\/[a-zA-Z0-9()@:%_+.~#?&//=]*)?/gi, tag: 'URL' },

  // 4. IBAN (Requires real ISO country code, eliminates Order IDs like OD438...)
  { pattern: new RegExp(String.raw`\b(?:${IBAN_COUNTRIES})\d{2}(?:[\s-]?\d{4}){3,7}\b`, 'gi'), tag: 'IBAN' },

  // 5. Indian Aadhaar Card (12 digits) & Masked
  { pattern: /\b[2-9]\d{3}[ -]\d{4}[ -]\d{4}\b/g, tag: 'AADHAAR' },
  { pattern: /\b[X]{4}[ -][X]{4}[ -]\d{4}\b/g, tag: 'AADHAAR' },

  // 6. Indian PAN Card
  { pattern: /\b[A-Z]{5}\d{4}[A-Z]\b/g, tag: 'PAN' },

  // 7. Indian TAN (Tax Deduction Account Number)
  { pattern: /\b[A-Z]{4}\d{5}[A-Z]\b/g, tag: 'TAN' },

  // 8. Indian GSTIN
  { pattern: /\b\d{2}[A-Z]{5}\d{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b/g, tag: 'GSTIN' },

  // 9. Indian Bank IFSC Code
  { pattern: /\b[A-Z]{4}0[A-Z0-9]{6}\b/g, tag: 'IFSC' },

  // 10. Indian Voter ID
  { pattern: /\b[A-Z]{3}\d{7}\b/g, tag: 'VOTER_ID' },

  // 11. Indian Driving License
  { pattern: /\b[A-Z]{2}[- ]?\d{2}[- ]?(?:19|20)\d{2}[- ]?\d{7}\b/g, tag: 'DRIVING_LICENSE' },

  // 12. Student & Medical Reference Tokens (Excludes generic corporate tokens)
  { pattern: /\b(?:AUS-STU-[\w-]+|MRN-[\w-]+|TX-MED-[\w-]+)\b/g, tag: 'ID' },

  // 13. Vehicle Registration (RC & Bharat Series BH)
  { pattern: /\b[A-Z]{2}[- ]?\d{1,2}[- ]?(?:[A-Z]{1,3}[- ]?)?\d{4}\b|\b\d{2}\s?BH\s?\d{4}\s?[A-Z]{1,2}\b/g, tag: 'VEHICLE_RC' },

  // 14. US Social Security Number (SSN) & ITIN
  { pattern: /\b\d{3}-\d{2}-\d{4}\b/g, tag: 'SSN' },
  { pattern: /\b9\d{2}-\d{2}-\d{4}\b/g, tag: 'ITIN' },

  // 15. UK National Insurance Number (NINO)
  { pattern: /\b[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]\b/gi, tag: 'UK_NINO' },

  // 16. Belgian National ID / BIS
  { pattern: /\b\d{2}\.\d{2}\.\d{2}-\d{3}\.\d{2}\b/g, tag: 'NATIONAL_ID' },

  // 17. Passports (Indian 8-char, US/Global alphanumeric)
  { pattern: /\b[A-PR-WYa-pr-wy][1-9]\d{6}\b|\b[A-Z]\d{7,8}[A-Z]?\b/g, tag: 'PASSPORT' },

  // 18. Credit & Debit Cards
  { pattern: /\b(?:4\d{3}|5[1-5]\d{2}|6011)[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b|\b3[47]\d{2}[-\s]?\d{6}[-\s]?\d{5}\b/g, tag: 'CARD' },

  // 19. Phone Numbers: Indian Mobile (6-9 followed by 9 digits)
  { pattern: /(?:\+91[\s.-]?|0091[\s.-]?)[6-9]\d{4}[\s.-]?\d{5}\b|\b0[6-9]\d{9}\b|\b[6-9]\d{9}\b/g, tag: 'PHONE' },
  // 20. Phone Numbers: Indian Landlines
  { pattern: /\b0(?:11|22|33|44|80)[- ]?\d{4}[- ]?\d{4}\b|\b0\d{3,4}[- ]?\d{6,7}\b/g, tag: 'PHONE' },
  // 21. Phone Numbers: US & International
  { pattern: /\+\d{1,3}[\s.-]\d{1,4}(?:[\s.-]\d{2,4}){2,4}\b|\+\d{1,3}[\s.-]\(?\d{2,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}\b|\(\d{3}\)[\s.-]?\d{3}[\s.-]?\d{4}\b/g, tag: 'PHONE' },

  // 22. Cryptographic Hashes
  { pattern: /\b[a-f0-9]{64}\b|\b[a-f0-9]{32}\b/gi, tag: 'HASH' },

  // 23. PGP Fingerprints
  { pattern: /(?:\b[0-9A-F]{4}\s*){8,10}\b/g, tag: 'KEY' },

  // 24. IP Addresses (IPv4)
  { pattern: /\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b/g, tag: 'IP_ADDRESS' },

  // 25. MAC Addresses
  { pattern: /\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b/g, tag: 'MAC_ADDRESS' },

  // 26. Bank Account Numbers (requires account context indicator to avoid matching order IDs/barcodes)
  { pattern: /\b(?:Account\s*(?:Number|No\.?|#)?|A\/c\s*(?:No\.?|#)?|Bank\s*A\/c|Acc\s*No\.?)\s*[:-]?\s*([0-9]{9,18})\b/gi, tag: 'ACCOUNT_NUMBER' },

  // 27. Indian Address Relations, Landmarks & Rural Markers (Strict word boundaries to never match "policy")
  { pattern: /\b(?:S\/o|D\/o|W\/o|C\/o|Son of|Daughter of|Wife of|Care of)\s*[:-]?\s*[A-Za-z][a-zA-Z.\s]{2,40}\b/gi, tag: 'ADDRESS' },
  { pattern: /\b(?:Near|Opposite|Behind|Beside|Next to|Adjacent to|Above|Below)\s*[:-]?\s*[A-Za-z][a-zA-Z.\s]{2,40}\b/gi, tag: 'ADDRESS' },
  { pattern: /\b(?:Vill(?:age)?\b|P\.O\.\b|Post\s*Office\b|Dist(?:rict)?\b|Taluk[a]?\b|Teh(?:sil)?\b|Mandal\b)\s*[:-]?\s*[A-Za-z][a-zA-Z.\s]{2,40}\b/gi, tag: 'ADDRESS' },

  // 28. Informal Indian Housing Prefixes (Room, Chawl, Gali)
  { pattern: /\b(?:Room|Chawl|Gali|House|Shop)\s*(?:No\.?|#|Number)?\s*[:-]?\s*[A-Z0-9/-]{1,10}\b/gi, tag: 'ADDRESS' },
];

// --- 2. CONTEXTUAL ADDRESS & BIRTHPLACE HEADERS ---
const ADDRESS_HEADER_REGEX = /(?:Current Residential Address|Residential Address|Assigned Workspace|Temporary Lodging|Prior Residential Address|Billing Address|Shipping Address|Mailing Address|Registered Office|Site Location|Delivery Address|Correspondence Address|Permanent Address)(?:\s*\([^)]*\))?:\s*\n?([^\n*#]+)/gi;
const BIRTH_HEADER_REGEX = /(?:Place of Birth):\s*([^\n*#]+)/gi;
const CAPS_NAME_HEADER_REGEX = /(?:Cardholder Name|Full Name|Name|Applicant Name|Patient Name|Student Name|Authorized Signatory|Father's Name|Spouse Name):\s*([A-Z]{2,}(?:\s+[A-Z]{2,})+)/gi;

// --- 3. UPGRADED COMPREHENSIVE ADDRESS VOCABULARIES ---
const esc = s => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

const STREET_TYPES = [
  'Street','St','Avenue','Ave','Boulevard','Blvd','Road','Rd','Drive','Lane','Ln','Way','Court','Ct',
  'Circle','Cir','Terrace','Ter','Parkway','Pkwy','Place','Pl','Square','Sq','Plaza','Loop','Row','Run',
  'Trail','Trl','Crossing','Xing','Crescent','Cres','Expressway','Expy','Junction','Jct','Point','Pt',
  'Ridge','Rdg','Valley','Vly','View','Vw','Vista','Walk','Broadway','Esplanade','Landing','Grove',
  'Heights','Hts','Hill','Hollow','Holw','Island','Isle','Meadow','Mews','Orchard','Park','Ranch','River',
  'Shore','Spring','Springs','Alley','Aly','Highway','Hwy','Freeway','Fwy','Causeway','Cswy','Bypass',
  'Byp','Extension','Ext','Path','Wharf','Quay',
  // India / South Asia
  'Marg','Salai','Bazaar','Bazar','Nagar','Colony','Sector','Phase','Enclave','Layout','Society','Vihar',
  'Peth','Chowk','Gali','Galli','Veedhi','Rasta',
];
const ST = STREET_TYPES.map(esc).join('|') + '|Dr';

const DIR       = 'N|S|E|W|NE|NW|SE|SW|North|South|East|West';
const UNITS     = 'Suite|Ste|Apt|Apartment|Unit|Floor|Fl|Office|Room|Rm|Flat|Plot|Tower|Block|Bldg|Building|Shop|Hse';
const US_STATES = 'AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC';
const COUNTRIES = [
  'United States of America','United States','United Kingdom','United Arab Emirates','South Korea',
  'South Africa','New Zealand','Sri Lanka','Saudi Arabia','India','USA','UK','UAE','Canada','Australia',
  'Germany','France','Netherlands','Belgium','Spain','Italy','Switzerland','Sweden','Singapore','Malaysia',
  'Indonesia','Philippines','Japan','China','Qatar','Kuwait','Oman','Bahrain','Nigeria','Kenya','Brazil',
  'Mexico','Argentina','Russia','Poland','Portugal','Ireland','Scotland','England','Vietnam','Thailand',
  'Pakistan','Bangladesh','Nepal','Maldives','Turkey','Egypt','Greece','Austria','Denmark','Norway','Finland'
].map(esc).join('|');

const UPGRADED_ADDRESS_PATTERNS = [
  // A. Number-first streets
  new RegExp(
    String.raw`\b(?!19\d{2}\b|20[0-2]\d\b)(?:` +
      String.raw`\d{1,6}(?:st|nd|rd|th)?[A-Za-z]?|` +
      String.raw`\d{1,5}-\d{1,4}|` +
      String.raw`\d{1,4}(?:\s?/\s?\d{1,4}){1,2}|` +
      String.raw`one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)` +
    String.raw`[\s,]+` +
    String.raw`(?:(?:${DIR})\.?\s+)?` +
    String.raw`(?:[A-Za-z0-9.'\-&]+\s+){1,5}` +
    String.raw`(?:${ST})\.?(?:\s+(?:${DIR})\.?)?\b` +
    String.raw`(?:[\s,#\-]*(?:${UNITS}|#)\s*[A-Za-z0-9/-]{1,10}\.?)?`,
    'gi'),

  // B. Unit-first
  new RegExp(
    String.raw`\b(?:${UNITS})\s*(?:No\.?)?\s*[:#]?\s*[A-Za-z0-9/-]{1,10}` +
    String.raw`[,\s]+(?:\w+\s+){0,3}(?:Rue|Avenue|Boulevard|Street|Road|Drive|Lane|Way|Allée|Platz|Str)\b[^,\n]+`,
    'gi'),

  // C. Indian Style
  new RegExp(
    String.raw`\b(?:Flat|Plot|House|Shop|Office|Suite|Apt|Door)\s*(?:No\.?|#)?\s*[\w\/-]+[,\s]+(?:[\w\s]+(?:Nagar|Colony|Sector|Phase|Enclave|Road|Marg|Street|Layout|Bazaar|Salai|Society|Apartments|Heights|Tower))\b`,
    'gi'),

  // D. European street-name-first
  (() => {
    const ends = ['straat','gracht','kade','laan','weg','dreef','singel','markt','plein','burgwal',
      'straße','strasse','gasse','platz','allee','damm','ring','promenade','boulevard','avenue','rue',
      'chemin','impasse','quai','route','allée','place','via','viale','corso','piazza','vicolo','calle',
      'carrera','paseo','glorieta','avinguda','rua','rúa','ulica','prospekt'];
    const E = [...ends, ...ends.map(w => w[0].toUpperCase() + w.slice(1))].map(esc).join('|');
    return new RegExp(String.raw`\b[A-Za-zà-ÿ'’\-]*(?:${E})\b(?:\s+[A-Za-zà-ÿ'’\-]+){0,4}\s+\d{1,4}[A-Za-z]?\b`, 'g');
  })(),

  // E. Indian "No./H.No./D.No." style
  /\b(?:H\.?\s?No\.?|D\.?\s?No\.?|Door\s?No\.?|Plot\s?No\.?|Shop\s?No\.?|Survey\s?No\.?|Sy\.?\s?No\.?)\s*[:.-]?\s*\d+(?:\s?[-/]\s?\d+){0,2}[A-Za-z]?\b/gi,

  // F. PO Boxes
  /\b(?:P\.?\s?O\.?|G\.?\s?P\.?\s?O\.?|Post(?:al)?)\s+Box\b(?:\s*(?:No\.?)?[:#-]?\s*[A-Z0-9-]{2,10}\b)?/gi,

  // G. Ordinal-only streets
  new RegExp(String.raw`\b\d{1,3}(?:st|nd|rd|th)\s+(?:${ST})\b(?:[\s,#\-]*(?:${UNITS}|#)\s*[A-Za-z0-9/-]{1,10})?`, 'gi'),

  // H. Named street without number
  new RegExp(String.raw`\b[A-Z][A-Za-z0-9.'&\-]{1,30}\s+(?:${ST})\.?\b`, 'g'),

  // I. Deterministic City + State / Country / PIN
  new RegExp(String.raw`\b[A-Z][a-zA-Z.'\-]+(?:\s+[A-Z][a-zA-Z.'\-]+){0,2},\s*(?:${US_STATES})\s+\d{5}(?:-\d{4})?\b`, 'g'),
  new RegExp(String.raw`\b[A-Z][a-zA-Z.'\-]+(?:\s+[A-Z][a-zA-Z.'\-]+){0,2}\s+(?:${US_STATES})\s+\d{5}(?:-\d{4})?\b`, 'g'),
  new RegExp(String.raw`\b[A-Z][a-zA-Z.'\-]+(?:\s+[A-Z][a-zA-Z.'\-]+){0,2},\s*(?:${US_STATES})\b`, 'g'),
  new RegExp(String.raw`\b[A-Z][a-zA-Z.'\-]+(?:\s+[A-Z][a-zA-Z.'\-]+){0,2},\s*(?:${COUNTRIES})\b`, 'g'),
  /\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2}\s+\d{6}\b/g,

  // J. Post codes & PIN codes
  /\b(?:zip(?:\s*code)?|pin(?:\s*code)?|pincode|postal(?:\s*code)?)\s*[:#-]?\s*[A-Z0-9][A-Z0-9-]{2,9}\b/gi,
  /\b(?:GIR\s?0AA|[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})\b/g, 
  /\b[ABCEGHJ-NPRSTVXY]\d[A-Z][ -]?\d[A-Z]\d\b/g,         
  /\b(?!(?:19|20)\d{2}\b)\d{4}[ \t]+(?=[A-Z][a-zA-Z])/g,  
  /\b[1-9]\d{2}\s?\d{3}\b/g,                              
  /\b\d{5}(?:-\d{4})?\b/g,                                

  // K. GPS Coordinates
  /[-+]?\d{1,2}\.\d{4,}\s*°?\s*[NSns]?\s*[,;]\s*[-+]?\d{1,3}\.\d{4,}\s*°?\s*[EWew]?/g,

  // L. Lowercase Indian Localities (meera nath nagar, juhu gully, hsr layout)
  /\b[a-zA-Z0-9.-]{2,20}\s+(?:nagar|colony|society|layout|vihar|enclave|gali|chawl|gully|peth|chowk)\b/gi,

  // M. Uncapitalized Indian City + State pairs (lucknow uttar pradesh, shahdara delhi)
  (() => {
    const IN_STATES = 'maharashtra|delhi|karnataka|tamil nadu|telangana|uttar pradesh|gujarat|rajasthan|west bengal|bihar|madhya pradesh|haryana|punjab|kerala|andhra pradesh|odisha|jharkhand|chhattisgarh|assam|goa|himachal pradesh|uttarakhand';
    return new RegExp(String.raw`\b(?:[a-zA-Z\-]+\s+){1,3}(?:${IN_STATES})\b`, 'gi');
  })()
];

function extractNERSpans(rawResults, line) {
  const entities = [];
  let current = null;

  for (const token of rawResults) {
    const isSubword = token.word.startsWith('##');
    const tagType = token.entity.replace(/^[BI]-/, '');

    const minScore = 0.80; 
    if (token.score < minScore) continue;

    if (current && (isSubword || (token.entity.startsWith('I-') && current.type === tagType))) {
      current.tokens.push(token);
    } else {
      if (current) entities.push(current);
      current = { type: tagType, tokens: [token] };
    }
  }
  if (current) entities.push(current);

  const spans = [];
  let searchIdx = 0;

  for (const ent of entities) {
    if (ent.type !== 'PER' && ent.type !== 'LOC') continue;

    const firstClean = ent.tokens[0].word.replace(/^##/, '');
    const lastClean = ent.tokens[ent.tokens.length - 1].word.replace(/^##/, '');

    const firstIdx = line.indexOf(firstClean, searchIdx);
    if (firstIdx === -1) continue;

    let lastIdx = line.indexOf(lastClean, firstIdx);
    if (lastIdx === -1) lastIdx = firstIdx;
    let endIdx = lastIdx + lastClean.length;

    if (ent.type === 'PER') {
      const remainder = line.substring(endIdx);
      const hyphenMatch = remainder.match(/^-[A-Z][a-zA-Z]+/);
      if (hyphenMatch) {
        endIdx += hyphenMatch[0].length;
      }
    }

    const extractedText = line.substring(firstIdx, endIdx).trim();
    const cleanLower = extractedText.toLowerCase().replace(/^[^\w]+|[^\w]+$/g, '');

    // Skip short tokens (< 3 chars), UI stopwords, strings without alphabets, or company/organization names
    if (
      cleanLower.length < 3 ||
      UI_STOPWORDS.has(cleanLower) ||
      !/[a-zA-Z]/.test(extractedText) ||
      COMPANY_INDICATORS.test(extractedText) ||
      COMPANY_INDICATORS.test(cleanLower)
    ) {
      searchIdx = endIdx;
      continue;
    }

    spans.push({
      tag: ent.type === 'PER' ? 'NAME' : 'LOCATION',
      text: extractedText,
      start: firstIdx,
      end: endIdx
    });

    searchIdx = endIdx;
  }

  return spans;
}

/**
 * Executes full hybrid PII Redaction pipeline on raw text.
 * @param {string} rawText - Input sensitive text
 * @returns {Promise<{ cleanedText: string, entities: Array, redactedEntities: Array, count: number, engine: string, timeTaken: number }>}
 */
export async function redactTextContent(rawText) {
  const startTime = performance.now();
  const { ner, engine } = await getNERPipeline();
  const redactions = [];

  const addSpan = (start, end, tag) => {
    const overlapIdx = redactions.findIndex(r => Math.max(start, r.start) < Math.min(end, r.end));
    if (overlapIdx !== -1) {
      const existing = redactions[overlapIdx];
      if (start <= existing.start && end >= existing.end && (start < existing.start || end > existing.end)) {
        redactions[overlapIdx] = { start, end, tag };
      }
      return;
    }
    redactions.push({ start, end, tag });
  };

  // --- PHASE 1: High-Precision Deterministic Regex Pass ---
  for (const rule of REGEX_RULES) {
    const pattern = new RegExp(rule.pattern.source, rule.pattern.flags);
    let m;
    while ((m = pattern.exec(rawText)) !== null) {
      addSpan(m.index, m.index + m[0].length, rule.tag);
    }
  }

  // --- PHASE 2: Contextual Address, Birthplace & Caps Name Headers ---
  let hm;
  while ((hm = ADDRESS_HEADER_REGEX.exec(rawText)) !== null) {
    const addr = hm[1].trim();
    const start = hm.index + hm[0].indexOf(addr);
    addSpan(start, start + addr.length, 'ADDRESS');
  }

  let bm;
  while ((bm = BIRTH_HEADER_REGEX.exec(rawText)) !== null) {
    const place = bm[1].trim();
    const start = bm.index + bm[0].indexOf(place);
    addSpan(start, start + place.length, 'LOCATION');
  }

  let cm;
  while ((cm = CAPS_NAME_HEADER_REGEX.exec(rawText)) !== null) {
    const nameStr = cm[1].trim();
    const start = cm.index + cm[0].indexOf(nameStr);
    addSpan(start, start + nameStr.length, 'NAME');
  }

  // --- PHASE 3: Upgraded Syntactic Address & Location Detection ---
  const NOT_CITY_PREFIX = /^(?:Invoice|Order|Case|Ticket|Ref|Reference|Reg|Roll|Account|Employee|ID|Transaction|Payment|Amount|Policy|Claim|Serial|Model|Part|Batch|Lot|Page|File|Document|Form|Row|Item|Code|Version|Chapter|Section|Table|Figure|Slide)\b/i;

  for (const pat of UPGRADED_ADDRESS_PATTERNS) {
    const pattern = new RegExp(pat.source, pat.flags);
    let m;
    while ((m = pattern.exec(rawText)) !== null) {
      const matchStr = m[0].trim();
      if (NOT_CITY_PREFIX.test(matchStr) && /\d{5,6}$/.test(matchStr)) {
        continue;
      }
      addSpan(m.index, m.index + m[0].length, 'ADDRESS');
    }
  }

  // --- PHASE 4: Semantic NER (Xenova BERT-NER) for Names & Locations ---
  const lines = rawText.split('\n');
  let lineOffset = 0;

  for (const line of lines) {
    const trimmed = line.trim();
    if (
      trimmed.length > 0 &&
      !trimmed.startsWith('---') &&
      !trimmed.startsWith('###') &&
      /[A-Z]/.test(trimmed) &&
      !SKIP_PATTERNS.test(trimmed)
    ) {
      const rawEnts = await ner(line);
      const nerSpans = extractNERSpans(rawEnts, line);
      for (const s of nerSpans) {
        addSpan(lineOffset + s.start, lineOffset + s.end, s.tag);
      }
    }
    lineOffset += line.length + 1; 
  }

  // --- PHASE 5: Sort, Synthesize & Merge Adjacent / Overlapping Spans ---
  redactions.sort((a, b) => a.start - b.start);

  const merged = [];
  for (const r of redactions) {
    if (merged.length === 0) {
      merged.push({ ...r });
    } else {
      const last = merged[merged.length - 1];
      const gap = rawText.substring(last.end, r.start);
      const isCleanSeparator = /^[,\s.\n\r\t-]*$/.test(gap);
      
      const isAddressMerge = (last.tag === 'ADDRESS' || last.tag === 'LOCATION') && 
                             (r.tag === 'ADDRESS' || r.tag === 'LOCATION');

      if (isAddressMerge && isCleanSeparator) {
        last.tag = 'ADDRESS';
        last.end = Math.max(last.end, r.end);
      } else if (!gap.includes('\n') && isCleanSeparator && gap.length <= 6 && last.tag === r.tag) {
        last.end = Math.max(last.end, r.end);
      } else {
        merged.push({ ...r });
      }
    }
  }

  // --- PHASE 6: Apply Redactions in Reverse Order ---
  let resultText = rawText;
  const redactedEntities = [];
  for (let i = merged.length - 1; i >= 0; i--) {
    const { start, end, tag } = merged[i];
    const originalValue = rawText.substring(start, end).trim();
    redactedEntities.unshift({ originalValue, tag, start, end });
    resultText = resultText.substring(0, start) + `[${tag}]` + resultText.substring(end);
  }

  const timeTaken = Math.round(performance.now() - startTime);

  return {
    cleanedText: resultText,
    entities: merged,
    redactedEntities,
    count: merged.length,
    engine,
    timeTaken
  };
}
