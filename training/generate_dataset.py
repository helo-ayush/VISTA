"""
High-Volume Synthetic Dataset Generator for Indian PII Redaction
Specialized exclusively for:
  - NAME: Full Indian & global human names across North, South, East, West, Central regions
  - ADDRESS: Full Indian & global addresses (House/Flat, Gali, Sector, Landmark, City, State, PIN)

Generates 100,000+ realistic examples with:
  - 40% Lowercase (ayush kumar, flat 402, rohini, delhi)
  - 30% UPPERCASE (AYUSH KUMAR, FLAT 402, ROHINI, DELHI)
  - 30% Title Case (Ayush Kumar, Flat 402, Rohini, Delhi)
  - Multi-document contexts (Invoices, IDs, KYC, Medical, Couriers, Chats)
  - Non-PII negative samples (Products, terms, UI text)
  - Exact character offsets and BIO token alignment
"""

import os
import json
import random
import re

# Set random seed for reproducibility
random.seed(42)

# =====================================================================
# 1. MASSIVE DEMOGRAPHIC INDIAN & GLOBAL NAMES REPOSITORY
# =====================================================================

INDIAN_FIRST_NAMES_MALE = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Aayan", "Krishna",
    "Ishan", "Shaurya", "Atharv", "Advik", "Pranav", "Advaith", "Aryan", "Dhruv", "Kabir",
    "Ritvik", "Darsh", "Kian", "Vedant", "Ayaan", "Rishaan", "Shlok", "Rohan", "Ayush",
    "Ansh", "Arman", "Dev", "Veer", "Aarush", "Rudra", "Harsh", "Rohit", "Suresh", "Ramesh",
    "Rajesh", "Deepak", "Manoj", "Sanjay", "Amit", "Rahul", "Vikas", "Sunil", "Anil", "Vijay",
    "Ajay", "Alok", "Ashish", "Gaurav", "Mayank", "Nitin", "Mohit", "Tarun", "Varun", "Chetan",
    "Dinesh", "Hemant", "Jagdish", "Lokesh", "Mukesh", "Naresh", "Pankaj", "Rakesh", "Sandeep",
    "Satish", "Umesh", "Vinod", "Yogesh", "Abhinav", "Abhishek", "Ankit", "Arvind", "Bhupendra",
    "Chandan", "Dharmendra", "Gajendra", "Jitendra", "Kuldeep", "Mahendra", "Narendra", "Ravindra",
    "Surendra", "Virendra", "Gurpreet", "Harpreet", "Manpreet", "Navdeep", "Paramveer", "Sukhwinder",
    "Balwinder", "Jaswinder", "Daljit", "Harjit", "Jagjit", "Manjit", "Ranjit", "Surjit", "Farhan",
    "Imran", "Salman", "Rehan", "Zeeshan", "Tariq", "Bilal", "Danish", "Faisal", "Hamza", "Irfan",
    "Junaid", "Nadeem", "Rizwan", "Sameer", "Suhail", "Tanveer", "Usman", "Wasim", "Zubair",
    "Karthik", "Venkatesh", "Srinivas", "Subramanian", "Murugan", "Senthil", "Praveen", "Mahesh",
    "Sudhir", "Satyanarayana", "Ranganathan", "Narayanan", "Gopalan", "Balaji", "Raghavan",
    "Anirudh", "Siddharth", "Tejas", "Chaitanya", "Shantanu", "Suyash", "Swapnil", "Omkar",
    "Tanmay", "Saurabh", "Chinmay", "Bhushan", "Avinash", "Prashant", "Sachin", "Nilesh",
    "Upnesh", "Jagannath", "Shravan", "Mukund", "Ramlal", "Sitaram", "Satyanarayan", "Gopal",
    "Brijesh", "Lalitesh", "Ramakant", "Chandresh", "Mithilesh", "Awadhesh", "Parmeshwar", "Bishnu"
]

INDIAN_FIRST_NAMES_FEMALE = [
    "Aanya", "Aadhya", "Aarohi", "Ananya", "Pari", "Anika", "Navya", "Angel", "Diya", "Myra",
    "Sara", "Siya", "Saanvi", "Avani", "Riya", "Kiara", "Shreya", "Ira", "Ahana", "Prisha",
    "Shanaya", "Kavya", "Manya", "Tanvi", "Vanya", "Ishita", "Ridhima", "Aditi", "Sneha",
    "Pooja", "Neha", "Priya", "Swati", "Shweta", "Divya", "Anjali", "Ritu", "Rekha", "Sunita",
    "Geeta", "Seema", "Meena", "Anita", "Asha", "Usha", "Sarita", "Kavita", "Sangeeta", "Manju",
    "Pushpa", "Lata", "Meera", "Radha", "Shanti", "Kamla", "Shakuntala", "Sushila", "Nirmala",
    "Savitri", "Harleen", "Simran", "Jasleen", "Kirandeep", "Maninder", "Gurleen", "Amanpreet",
    "Fatima", "Ayesha", "Zainab", "Sana", "Mariam", "Noor", "Bushra", "Farida", "Nasreen",
    "Parveen", "Shabana", "Tahira", "Yasmeen", "Deepika", "Kareena", "Katrina", "Shraddha",
    "Alia", "Anushka", "Madhuri", "Juhi", "Kajol", "Rani", "Vidya", "Kangana", "Taapsee",
    "Lakshmi", "Saraswati", "Parvati", "Gayatri", "Bhavani", "Meenakshi", "Soundarya", "Revathi",
    "Sujatha", "Padma", "Uma", "Vasantha", "Hemalatha", "Nandini", "Shubha", "Deepa", "Sandhya",
    "Madhu", "Kanti", "Urmila", "Lalita", "Manorama", "Champa", "Basanti", "Phoolo", "Kaushalya"
]

GLOBAL_FIRST_NAMES = [
    "John", "David", "Michael", "James", "Robert", "William", "Richard", "Thomas", "Charles",
    "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven", "Paul", "Andrew", "Joshua",
    "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth", "Barbara", "Susan", "Jessica", "Sarah",
    "Karen", "Nancy", "Lisa", "Betty", "Margaret", "Sandra", "Ashley", "Kimberly", "Emily", "Donna"
]

INDIAN_SURNAMES = [
    # North / Hindi Belt
    "Sharma", "Verma", "Gupta", "Singh", "Kumar", "Yadav", "Mishra", "Pandey", "Tiwari", "Shukla",
    "Dubey", "Chaubey", "Tripathi", "Pathak", "Joshi", "Bhatt", "Pant", "Upadhyay", "Dixit",
    "Awasthi", "Agrawal", "Bansal", "Mittal", "Garg", "Goyal", "Jindal", "Singhal", "Tayal",
    "Kansal", "Goel", "Bindal", "Khandelwal", "Maheshwari", "Somani", "Rathi", "Mundra", "Biyani",
    "Chauhan", "Rathore", "Tomar", "Sisodia", "Shekhawat", "Gehlot", "Bhati", "Solanki",
    # Maharashtra / Gujarat
    "Shinde", "Deshmukh", "Kulkarni", "Patil", "Pawar", "Jadhav", "Bhosale", "Gaikwad", "Chavan",
    "More", "Kadam", "Sawant", "Rane", "Salunkhe", "Mohite", "Thorat", "Jagtap", "Kamble",
    "Waghmare", "Sonawane", "Lokhande", "Patel", "Shah", "Mehta", "Desai", "Parikh", "Vora",
    "Modi", "Doshi", "Gandhi", "Trivedi", "Jani", "Dave", "Raval", "Vyas", "Thaker", "Pandya",
    # South India (Tamil Nadu, Karnataka, Andhra/Telangana, Kerala)
    "Narayanan", "Subramanian", "Venkatesh", "Venkataraman", "Ramachandran", "Krishnan",
    "Balakrishnan", "Ramanathan", "Swaminathan", "Natarajan", "Srinivasan", "Sundaram", "Iyer",
    "Iyengar", "Pillai", "Nair", "Menon", "Nambiar", "Kurup", "Warrier", "Namboodiri", "Panicker",
    "Reddy", "Rao", "Choudhury", "Chowdary", "Naidu", "Raju", "Varma", "Murthy", "Sastry",
    "Bhat", "Shetty", "Hegde", "Pai", "Kamath", "Prabhu", "Shenoy", "Nayak", "Alva", "Rai",
    # East India (Bengal, Odisha, Assam)
    "Chatterjee", "Banerjee", "Mukherjee", "Ganguly", "Bhattacharya", "Chakraborty", "Dasgupta",
    "Sengupta", "Ghosh", "Bose", "Mitra", "Dutta", "Roy", "Sen", "Das", "Pal", "Biswas", "Sarkar",
    "Mondal", "Bhowmick", "Majumdar", "Sikdar", "Halder", "Barman", "Saha", "Poddar", "Kundu",
    "Patnaik", "Mohanty", "Pradhan", "Behera", "Nayak", "Samal", "Swain", "Rout", "Bora", "Saikia",
    # Punjab / Sikh
    "Sandhu", "Sidhu", "Gill", "Dhillon", "Grewal", "Brar", "Mann", "Bajwa", "Cheema", "Chahal",
    "Randhawa", "Bains", "Deol", "Sahota", "Virk", "Dhaliwal", "Sekhon", "Pannu", "Toor", "Aulakh",
    # Muslim
    "Khan", "Ahmed", "Ali", "Sheikh", "Ansari", "Siddiqui", "Qureshi", "Malik", "Mirza", "Baig",
    "Sayed", "Kazmi", "Rizvi", "Naqvi", "Zaidi", "Usmani", "Farooqi", "Nomani", "Hashmi",
    # Bihar / Eastern / UP Rural & Regional
    "Mahto", "Mahato", "Paswan", "Mandal", "Prasad", "Thakur", "Kushwaha", "Manjhi", "Sah",
    "Sahu", "Keshri", "Chaudhary", "Choudhary", "Jha", "Mishra", "Pandey", "Tiwari", "Yadav",
    "Ram", "Raut", "Bhagat", "Khatun", "Khatoon", "Das", "Sahni", "Mukhiya", "Kumari",
    # Christian / Anglo-Indian
    "D'Souza", "Fernandes", "Pereira", "Rodrigues", "Pinto", "Lobo", "Sequeira", "Mascarenhas",
    "Cardozo", "Coutinho", "Britto", "Furtado", "Nazareth", "Saldanha", "Menezes", "Thomas",
    "Mathew", "Joseph", "George", "Philip", "Paul", "Alexander", "Jacob", "Varghese", "Kurian"
]

GLOBAL_SURNAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Wilson", "Anderson",
    "Taylor", "Thomas", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson", "White", "Harris"
]

ALL_FIRST_NAMES = INDIAN_FIRST_NAMES_MALE + INDIAN_FIRST_NAMES_FEMALE + GLOBAL_FIRST_NAMES
ALL_SURNAMES = INDIAN_SURNAMES + GLOBAL_SURNAMES

# =====================================================================
# 2. REALISTIC INDIAN & GLOBAL ADDRESS GEOGRAPHY
# =====================================================================

HOUSING_UNITS = [
    "Flat {num}", "Flat No. {num}", "Flat {alpha}-{num}", "Plot No. {num}", "Plot {num}",
    "House No. {num}", "H.No. {num}", "H.No. {num}/{sub}", "Door No. {num}", "D.No. {num}-{sub}",
    "Room No. {small}", "Bungalow No. {small}", "Villa {small}", "Survey No. {num}",
    "Khasra No. {num}", "Shop No. {small}", "Block {alpha}, Floor {small}", "Office No. {num}",
    "Tower {alpha}, Flat {num}", "Floor {small}, Building {alpha}"
]

HOUSING_NAMES = [
    "Green Valley Apartments", "Shanti Niketan", "Gokul Dham Society", "Surya Enclave",
    "Royal Palms", "Galaxy Heights", "Silver Oak Residency", "Sai Kripa Chawl",
    "Panchsheel Towers", "Gulmohar Apartments", "Prestige Falcon City", "Sobha Dream Acres",
    "Brigade Gateway", "Godrej Woods", "DLF The Aralias", "Hiranandani Gardens",
    "Lodha Bellissimo", "Tata Promont", "Rustomjee Seasons", "Purva Windermere",
    "Oberoi Splendor", "Mantri Square", "Salarpuria Sattva", "Aparna CyberLife",
    "My Home Bhooja", "Jayabheri Silicon County", "Kalpataru Radiance", "Runwal Bliss"
]

STREET_PATTERNS = [
    "Gali No. {small}", "Street No. {small}", "Lane {small}", "{small}th Cross", "{small}th Main",
    "MG Road", "Station Road", "Ring Road", "Link Road", "Bypass Road", "Church Road",
    "Temple Road", "College Road", "Hospital Road", "Market Road", "Gandhi Marg", "Subhash Marg",
    "Nehru Marg", "Tilak Marg", "Patel Marg", "Ambedkar Road", "Tagore Road", "Anna Salai",
    "Poonamallee High Road", "Mount Road", "Avinashi Road", "100 Feet Road", "80 Feet Road",
    "Double Road", "Pipeline Road", "Commercial Street", "Brigade Road", "Park Street",
    "Camac Street", "Gariahat Road", "Boring Road", "Bailey Road", "Ashok Marg", "Hazratganj"
]

LANDMARKS = [
    "Near Shiv Mandir", "Near Hanuman Mandir", "Near Kali Bari", "Near Gurudwara",
    "Near Jama Masjid", "Near St. Mary's Church", "Opp. Metro Station", "Opp. Metro Pillar {num}",
    "Behind Indian Oil Petrol Pump", "Behind Bharat Petroleum", "Near Railway Station",
    "Opp. Civil Hospital", "Near Central Park", "Behind State Bank of India", "Near Bus Stand",
    "Opp. Head Post Office", "Near Water Tank", "Beside City Mall", "Adjacent to Police Station",
    "Opposite D-Mart", "Near Apollo Pharmacy", "Behind Reliance Fresh", "Near City Center"
]

LOCALITY_CITY_PIN = [
    # Delhi NCR
    {"locality": "Rohini Sector 7", "city": "Delhi", "state": "Delhi", "pin": "110085"},
    {"locality": "Rohini Sector 15", "city": "Delhi", "state": "Delhi", "pin": "110089"},
    {"locality": "Dwarka Sector 10", "city": "New Delhi", "state": "Delhi", "pin": "110075"},
    {"locality": "Dwarka Sector 22", "city": "New Delhi", "state": "Delhi", "pin": "110077"},
    {"locality": "Janakpuri Block C", "city": "New Delhi", "state": "Delhi", "pin": "110058"},
    {"locality": "Pitampura", "city": "Delhi", "state": "Delhi", "pin": "110034"},
    {"locality": "Lajpat Nagar IV", "city": "New Delhi", "state": "Delhi", "pin": "110024"},
    {"locality": "Saket Block J", "city": "New Delhi", "state": "Delhi", "pin": "110017"},
    {"locality": "Vasant Kunj Sector B", "city": "New Delhi", "state": "Delhi", "pin": "110070"},
    {"locality": "Karol Bagh", "city": "New Delhi", "state": "Delhi", "pin": "110005"},
    {"locality": "Noida Sector 62", "city": "Noida", "state": "Uttar Pradesh", "pin": "201309"},
    {"locality": "Noida Sector 18", "city": "Noida", "state": "Uttar Pradesh", "pin": "201301"},
    {"locality": "Greater Noida Alpha 1", "city": "Greater Noida", "state": "Uttar Pradesh", "pin": "201310"},
    {"locality": "Indirapuram", "city": "Ghaziabad", "state": "Uttar Pradesh", "pin": "201014"},
    {"locality": "Vaishali Sector 4", "city": "Ghaziabad", "state": "Uttar Pradesh", "pin": "201010"},
    {"locality": "DLF Phase 3", "city": "Gurugram", "state": "Haryana", "pin": "122002"},
    {"locality": "Sushant Lok 1", "city": "Gurugram", "state": "Haryana", "pin": "122009"},
    {"locality": "Sohna Road Sector 48", "city": "Gurugram", "state": "Haryana", "pin": "122018"},

    # Bengaluru
    {"locality": "Indiranagar 100 Feet Road", "city": "Bengaluru", "state": "Karnataka", "pin": "560038"},
    {"locality": "Koramangala 4th Block", "city": "Bengaluru", "state": "Karnataka", "pin": "560034"},
    {"locality": "HSR Layout Sector 2", "city": "Bengaluru", "state": "Karnataka", "pin": "560102"},
    {"locality": "BTM Layout 2nd Stage", "city": "Bengaluru", "state": "Karnataka", "pin": "560076"},
    {"locality": "Whitefield ITPL Main Road", "city": "Bengaluru", "state": "Karnataka", "pin": "560066"},
    {"locality": "Electronic City Phase 1", "city": "Bengaluru", "state": "Karnataka", "pin": "560100"},
    {"locality": "Jayanagar 4th Block", "city": "Bengaluru", "state": "Karnataka", "pin": "560011"},
    {"locality": "JP Nagar 6th Phase", "city": "Bengaluru", "state": "Karnataka", "pin": "560078"},
    {"locality": "Marathahalli", "city": "Bengaluru", "state": "Karnataka", "pin": "560037"},
    {"locality": "Bellandur Outer Ring Road", "city": "Bengaluru", "state": "Karnataka", "pin": "560103"},

    # Mumbai / Pune
    {"locality": "Andheri East Chakala", "city": "Mumbai", "state": "Maharashtra", "pin": "400093"},
    {"locality": "Andheri West Lokhandwala", "city": "Mumbai", "state": "Maharashtra", "pin": "400053"},
    {"locality": "Bandra West Hill Road", "city": "Mumbai", "state": "Maharashtra", "pin": "400050"},
    {"locality": "Juhu Scheme", "city": "Mumbai", "state": "Maharashtra", "pin": "400049"},
    {"locality": "Borivali West IC Colony", "city": "Mumbai", "state": "Maharashtra", "pin": "400103"},
    {"locality": "Powai Hiranandani", "city": "Mumbai", "state": "Maharashtra", "pin": "400076"},
    {"locality": "Dadar West Shivaji Park", "city": "Mumbai", "state": "Maharashtra", "pin": "400028"},
    {"locality": "Thane West Ghodbunder Road", "city": "Thane", "state": "Maharashtra", "pin": "400607"},
    {"locality": "Vashi Sector 17", "city": "Navi Mumbai", "state": "Maharashtra", "pin": "400703"},
    {"locality": "Kothrud Paud Road", "city": "Pune", "state": "Maharashtra", "pin": "411038"},
    {"locality": "Hinjewadi Phase 1", "city": "Pune", "state": "Maharashtra", "pin": "411057"},
    {"locality": "Wakad Datta Mandir Road", "city": "Pune", "state": "Maharashtra", "pin": "411057"},
    {"locality": "Viman Nagar", "city": "Pune", "state": "Maharashtra", "pin": "411014"},
    {"locality": "Baner Pashan Link Road", "city": "Pune", "state": "Maharashtra", "pin": "411045"},

    # Hyderabad / Chennai
    {"locality": "Banjara Hills Road No 12", "city": "Hyderabad", "state": "Telangana", "pin": "500034"},
    {"locality": "Jubilee Hills Check Post", "city": "Hyderabad", "state": "Telangana", "pin": "500033"},
    {"locality": "Madhapur Cyber Towers", "city": "Hyderabad", "state": "Telangana", "pin": "500081"},
    {"locality": "Gachibowli Financial District", "city": "Hyderabad", "state": "Telangana", "pin": "500032"},
    {"locality": "Kukatpally Housing Board Colony", "city": "Hyderabad", "state": "Telangana", "pin": "500072"},
    {"locality": "Anna Nagar West 2nd Avenue", "city": "Chennai", "state": "Tamil Nadu", "pin": "600040"},
    {"locality": "T. Nagar Usman Road", "city": "Chennai", "state": "Tamil Nadu", "pin": "600017"},
    {"locality": "Adyar Gandhi Nagar", "city": "Chennai", "state": "Tamil Nadu", "pin": "600020"},
    {"locality": "Velachery 100 Feet Road", "city": "Chennai", "state": "Tamil Nadu", "pin": "600042"},
    {"locality": "Besant Nagar 4th Main", "city": "Chennai", "state": "Tamil Nadu", "pin": "600090"},

    # Kolkata / Other Major Cities
    {"locality": "Salt Lake Sector V", "city": "Kolkata", "state": "West Bengal", "pin": "700091"},
    {"locality": "New Town Action Area 1", "city": "Kolkata", "state": "West Bengal", "pin": "700156"},
    {"locality": "Ballygunge Circular Road", "city": "Kolkata", "state": "West Bengal", "pin": "700019"},
    {"locality": "Alipore Judges Court Road", "city": "Kolkata", "state": "West Bengal", "pin": "700027"},
    {"locality": "Malviya Nagar Calgiri Marg", "city": "Jaipur", "state": "Rajasthan", "pin": "302017"},
    {"locality": "Vaishali Nagar Amrapali Circle", "city": "Jaipur", "state": "Rajasthan", "pin": "302021"},
    {"locality": "Gomti Nagar Vibhuti Khand", "city": "Lucknow", "state": "Uttar Pradesh", "pin": "226010"},
    {"locality": "Hazratganj MG Marg", "city": "Lucknow", "state": "Uttar Pradesh", "pin": "226001"},
    {"locality": "Vijay Nagar AB Road", "city": "Indore", "state": "Madhya Pradesh", "pin": "452010"},
    {"locality": "Sector 17 Market", "city": "Chandigarh", "state": "Chandigarh", "pin": "160017"},
    {"locality": "Arera Colony E-6", "city": "Bhopal", "state": "Madhya Pradesh", "pin": "462016"},
    {"locality": "Boring Road Canal Road", "city": "Patna", "state": "Bihar", "pin": "800001"},
    {"locality": "Marine Drive Shanmugham Road", "city": "Kochi", "state": "Kerala", "pin": "682031"},
    {"locality": "RS Puram West DB Road", "city": "Coimbatore", "state": "Tamil Nadu", "pin": "641002"}
]

# =====================================================================
# 3. CONTEXTUAL DOCUMENT TEMPLATES
# =====================================================================

TEMPLATES = [
    # Invoices & Order Summaries
    "Billed To: {name}\nShipping Address: {address}\nOrder Total: ₹{amount}\nPayment Status: Confirmed",
    "Customer: {name}\nDelivery Location: {address}\nItem: {item}\nQty: 1",
    "Tax Invoice\nRecipient: {name}\nAddress: {address}\nInvoice No: INV-{num}",
    "Consignee: {name}\nDestination: {address}\nTracking ID: TRK{num}",
    "Deliver to {name} at {address} before {time}. Call on arrival.",
    "Order #{num} placed by {name}, will be delivered to {address}.",
    "Shipment for {name}, residing at {address}. Cash on Delivery.",

    # Official ID & KYC Forms
    "Applicant Name: {name}\nFather's Name: {relative}\nPermanent Address: {address}\nDOB: {dob}",
    "Cardholder: {name}\nAddress: {address}\nAccount Verified: Yes",
    "This is to certify that {name}, resident of {address}, has submitted KYC documents.",
    "Full Name: {name}\nResidential Address: {address}\nNationality: Indian",
    "Identity Card\nName: {name}\nS/o: {relative}\nAddress: {address}",
    "Patient: {name}\nAge: {age}\nAddress: {address}\nConsulting Doctor: Dr. {doctor}",

    # Healthcare / Prescriptions
    "Patient Name: {name}\nAddress: {address}\nDiagnosis: Routine Health Checkup\nRx: Paracetamol 500mg",
    "Medical Lab Report\nName: {name}\nReferred from: {address}\nTest: Complete Blood Count",

    # Delivery & Couriers
    "Urgent Courier for {name}, {address}. Please handle with care.",
    "Package dispatched to {name} at {address}.",
    "Delivery Attempted: Recipient {name} was not available at {address}.",

    # Conversational & Messages
    "Please send the contract to {name} at {address}.",
    "I have shared {name}'s contact address: {address}.",
    "Meeting confirmed with {name} at their office at {address}.",
    "The documents were signed by {name} who currently lives at {address}.",

    # Standalone Fields
    "Name: {name}\nAddress: {address}",
    "{name}, {address}",
    "Name: {name} | Location: {address}",

    # Academic & University Examination Templates
    "Examination Admit Card\nName: {name}\nRoll No. : {roll}\nInstitution : MAIT (Code: 148)\nProgramme : BTECH(CSE), Batch of: 2024, Code: 027",
    "GURU GOBIND SINGH INDRAPRASTHA UNIVERSITY, DELHI\nExamination Admit Card\nName: {name}\nRoll No: {roll}\nInstitution: MAIT (Code: 148)",
    "Student Identity Card\nName: {name}\nEnrollment No: {roll}\nDepartment: Computer Science & Engineering\nResidential Address: {address}",
    "Delhi Technological University (DTU)\nCandidate Name: {name}\nRoll No: {roll}\nPapers Appearing in: 027103(BS103) 027105(BS105)",
    "Provisional Degree Certificate\nThis is to certify that {name}, Roll No. {roll}, resident of {address}, has completed BTECH.",
    "Hall Ticket: {name}\nCenter: {address}\nReporting Time: 08:30 AM\nController of Examination",

    # Bank Passbook & Account Holder Templates
    "UTTAR BIHAR GRAMIN BANK\nName Of Account Holder : {name}\nFather's/Husband's Name : {relative}\nVillage: {address}\nAccount No:-...1000671030057180\nIFSC Code:- CBINOR10001",
    "Bank Passbook\nName Of Account Holder : {name}\nFather's/Husband's Name : {relative}\nPermanent Address: {address}\nBranch: Rural Development Branch",
    "STATE BANK OF INDIA\nAccount Holder Name: {name}\nS/o: {relative}\nAddress: {address}\nAccount Status: Active",
    "PUNJAB NATIONAL BANK\nCustomer Name: {name}\nC/o: {relative}\nCommunication Address: {address}",
    "Customer KYC Form\nPrimary Account Holder: {name}\nSpouse Name: {relative}\nResidential Address: {address}",

    # Board Marksheet & Certificate Templates
    "अंक पत्रक MARK SHEET\nSECONDARY SCHOOL EXAMINATION\nनाम Name: {name}\nमाता का नाम Mother's Name: {relative}\nपिता का नाम Father's Name: {doctor}\nविद्यालय School: OBC GIRL'S RESIDENTIAL HIGH SCHOOL\nरोल कोड Roll Code 51059\nरोल नं० Roll No. {roll}",
    "BIHAR SCHOOL EXAMINATION BOARD PATNA\nCandidate Name: {name}\nFather's Name: {relative}\nMother's Name: {doctor}\nBSEB Unique ID: 1232230590001\nExamination Center: {address}",
    "CENTRAL BOARD OF SECONDARY EDUCATION\nStudent Name: {name}\nMother's Name: {relative}\nFather's Name: {doctor}\nRoll No: {roll}",

    # Web Dashboard & User Management Table Templates
    "Test users\n{name}\nakshitamoghaa@gmail.com\nJoined Jan 29, 2026",
    "User Management Dashboard\nName: {name}\nEmail: dev_user_{num}@gmail.com\nLast signed in: Jan 29, 2026\nJoined: Jan 29, 2026",
    "Team Directory\n{name} | rustagiraghav@gmail.com | Last Active: Feb 14, 2025 | Location: {address}"
]

NON_PII_TEMPLATES = [
    "boAt Rockerz 450 Bluetooth On Ear Headphones with Mic, 40mm Dynamic Drivers, 15 Hours Playback, Matte Black",
    "Samsung Galaxy M14 5G Smoky Teal, 6GB RAM, 128GB Storage | 50MP Triple Cam | 6000mAh Battery",
    "Redmi 12 5G Jade Black 8GB RAM 256GB ROM Snapdragon 4 Gen 2 50MP AI Dual Camera",
    "Puma Men's Regular Fit Round Neck T-Shirt, Breathable Cotton Fabric, Navy Blue, Size Large",
    "Prestige Iris 750 Watt Mixer Grinder with 3 Stainless Steel Jars and 1 Juicer Jar, White and Blue",
    "Philips Multi Grooming Kit 9-in-1 Face, Hair and Body Trimmer for Men with DualCut Blades",
    "Terms and Conditions: All products sold are covered under a 7-day replacement policy.",
    "Return Policy: Items must be unused with original tags and packaging intact for return processing.",
    "Order confirmation email has been sent. Track your live shipment via the mobile application.",
    "Privacy Notice: Data processed in compliance with global security regulations and encryption standards.",
    "Customer Support is available 24/7. Call toll free 1800-200-3000 for order inquiries.",
    "Specifications: Dimensions 145 x 70 x 8 mm, Weight 175 grams, Li-Po 5000 mAh non-removable battery.",

    # Calendar Dates, Months & Timestamps (Must NEVER be tagged as NAME)
    "Jan 29, 2026 Last signed in Jan 29, 2026 Joined",
    "Feb 14, 2025 Created at Feb 14, 2025 Updated at Feb 14, 2025",
    "Mar 15, 2024 Date of Verification: Mar 15, 2024",
    "Apr 01, 2026 Effective Date: Apr 01, 2026",
    "May 20, 2025 Published on: May 20, 2025",
    "Jun 30, 2026 Expiry Date: Jun 30, 2026",
    "Jul 04, 2025 Registered on: Jul 04, 2025",
    "Aug 15, 2026 Holiday Calendar: Aug 15, 2026",
    "Sep 05, 2024 Assessment Date: Sep 05, 2024",
    "Oct 31, 2025 Renewal Date: Oct 31, 2025",
    "Nov 14, 2024 Issue Date: Nov 14, 2024",
    "Dec 25, 2026 System Maintenance: Dec 25, 2026",
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec 2024 2025 2026 2027",
    "January February March April May June July August September October November December",
    "Monday Tuesday Wednesday Thursday Friday Saturday Sunday Mon Tue Wed Thu Fri Sat Sun",

    # Academic & University Examination Non-PII Templates
    "GURU GOBIND SINGH INDRAPRASTHA UNIVERSITY, DELHI Examination Admit Card COE/EXAM/PO1429/MAR.22",
    "Institution : MAIT (Code: 148) Programme : BTECH(CSE), Batch of: 2024, Code: 027",
    "Institution : DTU (Delhi Technological University) Department of Computer Science & Engineering",
    "Institution : NSUT (Netaji Subhas University of Technology) BTECH Information Technology",
    "Indian Institute of Technology Delhi (IITD) Hauz Khas New Delhi Semester Examination",
    "Papers Appearing in : 027103(BS103) 027105(BS105) 027107(ES107) 027111(BS111) 027151(BS151)",
    "Course Structure: CS101 Introduction to Computing, MA102 Linear Algebra, PH103 Electromagnetism",
    "Entry Start : 45 Minutes before the commencement of the exam. Gate Closing Time : 15 Minutes.",
    "Controller of Examination, Dean Academic Affairs, Principal & Director of Institution",
    "Instructions to Candidates: Bags, mobile phones, and programmable calculators are strictly prohibited.",
    "Hall Ticket Valid for Mid-Term Assessment 2024-2025. Verify all course codes carefully.",
    "Office of the Controller of Examinations: Gazette Result Notification No. COE/RES/2024/091",

    # Board Marksheet Subjects, Marks, Labels & Officers (Must NEVER be tagged as NAME)
    "अंक पत्रक MARK SHEET SECONDARY SCHOOL EXAMINATION 2024 (ANNUAL)",
    "SUBJECT MARKS OBTAINED THEORY INTERNAL ASSESSMENT TOTAL MARKS IN WORDS",
    "MIL-HINDI 100 30 091 091 NINETY ONE",
    "SIL-SANSKRIT 100 30 084 084 EIGHTY FOUR",
    "MATHEMATICS 100 30 087 087 EIGHTY SEVEN",
    "SCIENCE 100 30 063 019 082 EIGHTY TWO",
    "SOCIAL SCIENCE 100 30 064 009 082 EIGHTY TWO",
    "ENGLISH 100 30 053 053 FIFTY THREE",
    "PHYSICS CHEMISTRY BIOLOGY HISTORY GEOGRAPHY ECONOMICS CIVICS COMPUTER SCIENCE",
    "AGGREGATE 426 RESULT 1ST DIV. PASS MARKS 150 TOTAL 500 RESULT: PASS",
    "रोल कोड Roll Code 51059 रोल नं० Roll No. 2400001 51059-00001-23",
    "बी.एस.ई.बी.यूनिक आई.डी. BSEB Unique ID 1232230590001 विद्यालय School",
    "Patna, Dated : 31/03/2024 परीक्षा नियंत्रक Controller of Examination",
    "Signature & Seal of Centre Superintendent Controller of Examination BSEB Patna",

    # Bank Passbook, Branch & Kiosk Non-PII Templates
    "UTTAR BIHAR GRAMIN BANK Sponsored by Central Bank of India",
    "Sunahra Spana Kendra- Sutihara Ram (9586) Link Branch- Surasand",
    "Customer Service Point (CSP) Kiosk Banking Business Correspondent (BC)",
    "Signature & Seal of B.C / Branch Manager / Authorized Officer",
    "Account No:-...1000671030057180 IFSC Code:- CBINOR10001 Aadhar No:- Mobile No:-",
    "State:-BIHAR PIN:- Branch Code: 9586 CIF No: 8829104",
    "STATE BANK OF INDIA PUNJAB NATIONAL BANK BANK OF BARODA CANARA BANK HDFC BANK ICICI BANK",

    # Web Dashboard & UI Table Non-PII Templates
    "Test users Last signed in Joined Actions Status Role",
    "Use this development instance for internal and test users. When you're ready to invite real users, create your production instance.",
    "Search users by name or email... Create user Filter Sort Export Download"
]

ITEMS = [
    "Wireless Earbuds", "Smartwatch with AMOLED Display", "Running Shoes",
    "Men's Casual Shirt", "Formal Leather Shoes", "Stainless Steel Water Bottle",
    "Mechanical Gaming Keyboard", "Ergonomic Office Chair", "Cotton Bedsheet Queen Size",
    "Induction Cooktop 2000W", "Electric Kettle 1.5L", "Backpack for 15.6 inch Laptop"
]

# =====================================================================
# 4. SYNTHESIS ENGINE
# =====================================================================

def generate_random_name():
    first = random.choice(ALL_FIRST_NAMES)
    last = random.choice(ALL_SURNAMES)
    if random.random() < 0.15:
        # Include middle name or initial (e.g. Aarav K. Sharma)
        middle = random.choice(INDIAN_FIRST_NAMES_MALE)
        if random.random() < 0.5:
            return f"{first} {middle[0]}. {last}"
        return f"{first} {middle} {last}"
    return f"{first} {last}"

def generate_random_address():
    loc_entry = random.choice(LOCALITY_CITY_PIN)
    parts = []

    # 1. Housing number/unit (85% probability)
    if random.random() < 0.85:
        pattern = random.choice(HOUSING_UNITS)
        housing = pattern.format(
            num=random.randint(1, 999),
            alpha=random.choice(["A", "B", "C", "D", "E"]),
            small=random.randint(1, 30),
            sub=random.randint(1, 4)
        )
        parts.append(housing)

    # 2. Apartment / Society name (45% probability)
    if random.random() < 0.45:
        parts.append(random.choice(HOUSING_NAMES))

    # 3. Street / Road (60% probability)
    if random.random() < 0.60:
        st_pattern = random.choice(STREET_PATTERNS)
        parts.append(st_pattern.format(small=random.randint(1, 20)))

    # 4. Landmark (40% probability)
    if random.random() < 0.40:
        lm_pattern = random.choice(LANDMARKS)
        parts.append(lm_pattern.format(num=random.randint(10, 400)))

    # 5. Locality
    parts.append(loc_entry["locality"])

    # 6. City, State, PIN
    r = random.random()
    if r < 0.70:
        parts.append(f"{loc_entry['city']}, {loc_entry['state']} {loc_entry['pin']}")
    elif r < 0.90:
        parts.append(f"{loc_entry['city']} {loc_entry['pin']}")
    else:
        parts.append(f"{loc_entry['city']}, {loc_entry['state']}")

    separator = random.choice([", ", ", ", ",\n", " - ", " "])
    return separator.join(parts)

def apply_casing(text, spans, mode):
    """
    Transforms text and span offsets according to casing mode:
      'lower' -> all lowercase
      'upper' -> all UPPERCASE
      'title' -> natural Title Case (unchanged)
    """
    if mode == "lower":
        new_text = text.lower()
    elif mode == "upper":
        new_text = text.upper()
    else:
        new_text = text
    return new_text, spans

def generate_sample():
    # 18% probability of pure non-PII negative text (Dates, Subjects, UI tables, Bank headers, Product specs)
    if random.random() < 0.18:
        text = random.choice(NON_PII_TEMPLATES)
        casing_mode = random.choices(["lower", "upper", "title"], weights=[0.4, 0.3, 0.3])[0]
        if casing_mode == "lower":
            text = text.lower()
        elif casing_mode == "upper":
            text = text.upper()
        return {
            "text": text,
            "spans": []
        }

    template = random.choice(TEMPLATES)
    name = generate_random_name()
    address = generate_random_address()
    relative = generate_random_name()
    doctor = generate_random_name()

    # Fill non-PII placeholders
    formatted = template.format(
        name="__NAME_PLACEHOLDER__",
        address="__ADDRESS_PLACEHOLDER__",
        relative="__REL_PLACEHOLDER__",
        doctor="__DOC_PLACEHOLDER__",
        amount=random.randint(499, 25000),
        item=random.choice(ITEMS),
        num=random.randint(1000, 99999),
        roll=f"{random.randint(100, 999)}{random.randint(10000000, 99999999)}",
        time=f"{random.randint(1, 8)} PM",
        dob=f"{random.randint(1, 28)}/0{random.randint(1, 9)}/{random.randint(1975, 2005)}",
        age=random.randint(18, 70)
    )

    # Replace placeholders and calculate exact start/end offsets
    spans = []

    # 1. Main Name
    if "__NAME_PLACEHOLDER__" in formatted:
        idx = formatted.index("__NAME_PLACEHOLDER__")
        formatted = formatted[:idx] + name + formatted[idx + len("__NAME_PLACEHOLDER__"):]
        spans.append({"start": idx, "end": idx + len(name), "label": "NAME"})

    # 2. Relative Name
    if "__REL_PLACEHOLDER__" in formatted:
        idx = formatted.index("__REL_PLACEHOLDER__")
        formatted = formatted[:idx] + relative + formatted[idx + len("__REL_PLACEHOLDER__"):]
        spans.append({"start": idx, "end": idx + len(relative), "label": "NAME"})

    # 3. Doctor Name
    if "__DOC_PLACEHOLDER__" in formatted:
        idx = formatted.index("__DOC_PLACEHOLDER__")
        formatted = formatted[:idx] + doctor + formatted[idx + len("__DOC_PLACEHOLDER__"):]
        spans.append({"start": idx, "end": idx + len(doctor), "label": "NAME"})

    # 4. Address
    if "__ADDRESS_PLACEHOLDER__" in formatted:
        idx = formatted.index("__ADDRESS_PLACEHOLDER__")
        formatted = formatted[:idx] + address + formatted[idx + len("__ADDRESS_PLACEHOLDER__"):]
        spans.append({"start": idx, "end": idx + len(address), "label": "ADDRESS"})

    # Sort spans by start index
    spans.sort(key=lambda s: s["start"])

    # Apply casing distribution: 40% lowercase, 30% UPPERCASE, 30% Title Case
    casing_mode = random.choices(["lower", "upper", "title"], weights=[0.4, 0.3, 0.3])[0]
    final_text, final_spans = apply_casing(formatted, spans, casing_mode)

    return {
        "text": final_text,
        "spans": final_spans
    }

def bio_tokenize(text, spans):
    """
    Converts raw text and character spans into word tokens and BIO labels:
      ['O', 'B-NAME', 'I-NAME', 'O', 'B-ADDR', 'I-ADDR', ...]
    """
    # Simple regex word/punctuation tokenization
    token_matches = list(re.finditer(r'\S+', text))
    tokens = [m.group(0) for m in token_matches]
    tags = ["O"] * len(tokens)

    for span in spans:
        span_start = span["start"]
        span_end = span["end"]
        label = span["label"]

        first_token = True
        for i, match in enumerate(token_matches):
            t_start, t_end = match.span()
            # If token overlaps with the span
            if max(t_start, span_start) < min(t_end, span_end):
                if first_token:
                    tags[i] = f"B-{label}"
                    first_token = False
                else:
                    tags[i] = f"I-{label}"

    return tokens, tags

# =====================================================================
# 5. DATASET GENERATION MAIN
# =====================================================================

def main():
    TOTAL_SAMPLES = 100_000
    VAL_RATIO = 0.05  # 5,000 validation samples, 95,000 training samples

    print(f"================================================================")
    print(f"Generating {TOTAL_SAMPLES:,} Authentic Indian & Global PII Records...")
    print(f"Target Labels: NAME and ADDRESS")
    print(f"Casing Distribution: 40% lowercase, 30% UPPERCASE, 30% Title Case")
    print(f"================================================================")

    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "val.jsonl")

    val_count = int(TOTAL_SAMPLES * VAL_RATIO)
    train_count = TOTAL_SAMPLES - val_count

    name_tags_count = 0
    addr_tags_count = 0

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val:

        for i in range(TOTAL_SAMPLES):
            sample = generate_sample()
            tokens, tags = bio_tokenize(sample["text"], sample["spans"])

            record = {
                "id": i,
                "text": sample["text"],
                "spans": sample["spans"],
                "tokens": tokens,
                "ner_tags": tags
            }

            line = json.dumps(record, ensure_ascii=False) + "\n"

            if i < train_count:
                f_train.write(line)
            else:
                f_val.write(line)

            for span in sample["spans"]:
                if span["label"] == "NAME":
                    name_tags_count += 1
                elif span["label"] == "ADDRESS":
                    addr_tags_count += 1

            if (i + 1) % 10_000 == 0 or (i + 1) == TOTAL_SAMPLES:
                print(f"[{i+1:,}/{TOTAL_SAMPLES:,}] samples generated... ({name_tags_count:,} Names, {addr_tags_count:,} Addresses)")

    print("\n[SUCCESS] Dataset Generation Complete!")
    print(f"  Training File:   {train_path} ({train_count:,} samples)")
    print(f"  Validation File: {val_path} ({val_count:,} samples)")
    print(f"  Total Names:     {name_tags_count:,}")
    print(f"  Total Addresses: {addr_tags_count:,}")

if __name__ == "__main__":
    main()
