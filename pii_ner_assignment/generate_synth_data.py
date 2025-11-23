import json
import random
from pathlib import Path


# ----- GLOBAL SETTINGS -----
N_TRAIN = 400
N_DEV = 80
N_TEST = 80


PII_NAMES = [
    "ramesh sharma", "priyanka verma", "rohan mehta", "sneha patel",
    "amit kumar", "anita singh", "karan joshi", "neha gupta"
]

CITIES = ["mumbai", "delhi", "chennai", "bangalore", "kolkata", "pune"]
DOMAINS = ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
DATES = ["01 02 2024", "15 08 2025", "31 12 2023", "05 01 2026", "09 09 2027"]

FILLERS = [
    "uh yeah so yesterday I was trying to check my balance but the network kept dropping and I couldn't really understand what was happening on the screen",
    "you know actually I was talking to someone earlier about the same issue and they told me to restart my phone but that didn't fix anything at all",
    "so basically for the past few days the connection has been super slow and it's getting really inconvenient because I need it for my work every single day",
    "i mean I tried logging in multiple times and every time it just shows some random error code that I honestly have no idea how to deal with",
    "like you know I was supposed to complete an important payment yesterday but the bank server kept timing out again and again for no clear reason",
    "actually when I called earlier the executive told me that the system was under maintenance but even now it seems to be behaving in the same strange way",
    "i was thinking that maybe it's a problem with the app version I'm using because it hasn't updated in a while and could be causing all these issues",
    "so listen I was travelling last week and the network there was terrible so that might have messed something up with my settings but I'm not really sure",
    "uh I remember that once before something similar happened and the only thing that worked was reinstalling the whole application from scratch which was annoying",
    "you know how sometimes the system just freezes for no reason right well that's exactly what it's been doing whenever I open the account details section",
    "basically I want to clarify that I didn't change any settings at all everything was the same as before but suddenly things stopped working properly",
    "i mean it's not like I'm trying something complicated it's just simple login and password but still it keeps rejecting it like it's not even my account",
    "I was trying to explain this to the previous agent but the call got disconnected midway and then I had to repeat the entire thing all over again",
    "so earlier today I tried from another device and the exact same issue happened which makes me think that it's definitely not a device related problem",
    "you know the strange part is that it works sometimes randomly for a few seconds and then suddenly stops again without giving any error message at all",
    "like honestly it's getting a bit frustrating now because I need quick access to my information and this delay is creating unnecessary problems for me",
    "actually I checked with a friend who uses the same service and they said everything is working perfectly fine for them so I don't get what's wrong",
    "uh also I noticed that some notifications aren't coming through properly and I have to refresh the app multiple times before anything shows up",
    "so what I'm trying to say is that I've already tried all the basic troubleshooting steps but the problem still persists without any improvement",
    "i mean at this point I'm just hoping someone can take a proper look into the issue because it's affecting things more than I initially assumed",
    "listen earlier when I logged in from my laptop the interface looked completely different and some options were missing which was really confusing",
    "you know there were moments when it seemed like the system was catching up and then it just froze again like it couldn't handle the request load",
    "basically I want to get this sorted before the end of the day because I have some deadlines and I can't afford these interruptions now",
    "so yesterday I tried contacting support but the queue was too long so I had to wait nearly twenty minutes before someone finally picked up the call",
    "uh the strange thing is that when I try using mobile data instead of wifi everything becomes even slower than usual which doesn't make sense",
    "i know these issues happen sometimes but it's been three days now and it's getting a bit too much since nothing is working consistently",
    "you know when the screen freezes it doesn't even let me close the app properly and I have to force quit it every single time which is annoying",
    "actually I noticed that transactions are also taking longer to process and sometimes they don't show up until several minutes which is concerning",
    "so I checked the FAQ page to see if there's any known outage or something but there was no update so I'm not sure what else to do right now",
    "uh before calling I also tried clearing the app cache and restarting the device but absolutely nothing changed in the performance or speed",
    "i really hope this gets resolved soon because I'm depending on this for an important task and delays like these make things really complicated",
    "listen I don't want to keep calling again and again so please help me understand what exactly is happening and how I can get it fixed quickly",
    "you know even simple things like opening recent activity take much longer than usual and that's making the entire experience unnecessarily slow",
    "so overall the whole service has been unstable and I'm honestly confused why it's acting like this because everything used to be smooth earlier"
]


# ----- HELPERS -----
def random_phone_digits():
    return "".join(str(random.randint(0, 9)) for _ in range(10))


def random_credit_card():
    blocks = ["".join(str(random.randint(0, 9)) for _ in range(4)) for _ in range(4)]
    return " ".join(blocks)


def make_email(name, domain):
    parts = name.split()
    local = f"{parts[0]} dot {parts[-1]}"
    domain_spoken = domain.replace(".", " dot ")
    return f"{local} at {domain_spoken}"


# ----- PII SENTENCE GENERATION -----
def generate_pii_sentence():
    name = random.choice(PII_NAMES)
    city = random.choice(CITIES)
    domain = random.choice(DOMAINS)
    date = random.choice(DATES)

    phone = random_phone_digits()
    card = random_credit_card()
    email = make_email(name, domain)

    # Pick one PII type at a time
    pii_type = random.choice(["NAME", "PHONE", "EMAIL", "CARD", "DATE", "CITY"])

    if pii_type == "NAME":
        return f"my name is {name}", [(name, "PERSON_NAME")]

    if pii_type == "PHONE":
        return f"my number is {phone}", [(phone, "PHONE")]

    if pii_type == "EMAIL":
        return f"my email id is {email}", [(email, "EMAIL")]

    if pii_type == "CARD":
        return f"my credit card number is {card}", [(card, "CREDIT_CARD")]

    if pii_type == "DATE":
        return f"i will travel on {date}", [(date, "DATE")]

    if pii_type == "CITY":
        return f"i live in {city}", [(city, "CITY")]


# ----- FILLER SENTENCE GENERATION -----
def generate_filler():
    return random.choice(FILLERS)


# ----- BUILD FULL TRANSCRIPT -----
def build_example(idx):
    # Generate 1 PII sentence
    pii_sentence, pii_items = generate_pii_sentence()

    # Generate 1–3 filler sentences
    fillers = [generate_filler() for _ in range(random.randint(1, 3))]

    # Choose random insertion position
    insert_pos = random.randint(0, len(fillers))

    # Build final list
    sentences = fillers[:insert_pos] + [pii_sentence] + fillers[insert_pos:]

    # Join with spaces to form final transcript
    text = " ".join(sentences)

    # Build entity spans
    entities = []
    for substring, label in pii_items:
        start = text.index(substring)
        end = start + len(substring)
        entities.append({"start": start, "end": end, "label": label})

    uid = f"utt_{idx:04d}"
    return uid, text, entities


# ----- WRITE FILES -----
def write_jsonl(path: Path, n: int, start_index: int):
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            uid, text, entities = build_example(start_index + i)
            obj = {"id": uid, "text": text, "entities": entities}
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def main():
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    write_jsonl(data_dir / "train.jsonl", N_TRAIN, 0)
    write_jsonl(data_dir / "dev.jsonl", N_DEV, 10000)
    write_jsonl(data_dir / "test.jsonl", N_TEST, 20000)

    print("Synthetic multi-sentence PII dataset created.")


if __name__ == "__main__":
    main()
