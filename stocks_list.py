import requests
import pandas as pd
import pickle
import os
import time

def get_all_nse_stocks():
    print("Fetching all NSE stocks from NSE India official website...")

    headers = {
        'User-Agent'     : 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept'         : 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection'     : 'keep-alive',
    }

    # NSE publishes full equity list publicly as CSV
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    try:
        session = requests.Session()
        # Hit homepage first to get cookies
        session.get("https://www.nseindia.com", headers=headers, timeout=15)
        time.sleep(2)

        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        from io import StringIO
        df = pd.read_csv(StringIO(response.text))
        df.columns = df.columns.str.strip()

        print(f"Raw NSE list: {len(df)} stocks")
        print(f"Columns: {df.columns.tolist()}")

        stock_map = {}
        for _, row in df.iterrows():
            sym  = str(row['SYMBOL']).strip()
            name = str(row['NAME OF COMPANY']).strip()
            if sym and sym != 'nan':
                stock_map[sym] = {
                    'symbol'    : sym,
                    'yf_symbol' : f"{sym}.NS",
                    'name'      : name,
                }

        print(f"Total valid stocks: {len(stock_map)}")
        return stock_map

    except Exception as e:
        print(f"NSE website fetch failed: {e}")
        print("Trying backup method...")
        return get_backup_list()


def get_backup_list():
    """
    Backup: hardcoded comprehensive list of 400+ major NSE stocks
    covering Nifty 50, Nifty Next 50, Nifty Midcap 150, popular small caps
    """
    print("Using comprehensive backup list...")

    stocks = {
        # ── Nifty 50 ──────────────────────────────────────────
        "RELIANCE"   : "Reliance Industries",
        "TCS"        : "Tata Consultancy Services",
        "HDFCBANK"   : "HDFC Bank",
        "INFY"       : "Infosys",
        "ICICIBANK"  : "ICICI Bank",
        "HINDUNILVR" : "Hindustan Unilever",
        "SBIN"       : "State Bank of India",
        "BAJFINANCE" : "Bajaj Finance",
        "BHARTIARTL" : "Bharti Airtel",
        "KOTAKBANK"  : "Kotak Mahindra Bank",
        "LT"         : "Larsen & Toubro",
        "WIPRO"      : "Wipro",
        "AXISBANK"   : "Axis Bank",
        "MARUTI"     : "Maruti Suzuki",
        "SUNPHARMA"  : "Sun Pharmaceutical",
        "TITAN"      : "Titan Company",
        "ULTRACEMCO" : "UltraTech Cement",
        "ASIANPAINT" : "Asian Paints",
        "NESTLEIND"  : "Nestle India",
        "POWERGRID"  : "Power Grid Corporation",
        "NTPC"       : "NTPC",
        "TECHM"      : "Tech Mahindra",
        "HCLTECH"    : "HCL Technologies",
        "BAJAJFINSV" : "Bajaj Finserv",
        "ONGC"       : "Oil & Natural Gas",
        "COALINDIA"  : "Coal India",
        "JSWSTEEL"   : "JSW Steel",
        "TATASTEEL"  : "Tata Steel",
        "ADANIENT"   : "Adani Enterprises",
        "ADANIPORTS" : "Adani Ports",
        "GRASIM"     : "Grasim Industries",
        "DIVISLAB"   : "Divi's Laboratories",
        "DRREDDY"    : "Dr. Reddy's Laboratories",
        "CIPLA"      : "Cipla",
        "EICHERMOT"  : "Eicher Motors",
        "BPCL"       : "Bharat Petroleum",
        "HEROMOTOCO" : "Hero MotoCorp",
        "HINDALCO"   : "Hindalco Industries",
        "INDUSINDBK" : "IndusInd Bank",
        "TATAMOTORS" : "Tata Motors",
        "SBILIFE"    : "SBI Life Insurance",
        "HDFCLIFE"   : "HDFC Life Insurance",
        "BRITANNIA"  : "Britannia Industries",
        "TATACONSUM" : "Tata Consumer Products",
        "ITC"        : "ITC",
        "UPL"        : "UPL",
        "BAJAJ-AUTO" : "Bajaj Auto",
        "APOLLOHOSP" : "Apollo Hospitals",
        "ADANIGREEN" : "Adani Green Energy",
        "SHRIRAMFIN" : "Shriram Finance",
        # ── Nifty Next 50 ─────────────────────────────────────
        "BANKBARODA" : "Bank of Baroda",
        "CANBK"      : "Canara Bank",
        "PNB"        : "Punjab National Bank",
        "IDFCFIRSTB" : "IDFC First Bank",
        "FEDERALBNK" : "Federal Bank",
        "BANDHANBNK" : "Bandhan Bank",
        "RBLBANK"    : "RBL Bank",
        "YESBANK"    : "Yes Bank",
        "LICHSGFIN"  : "LIC Housing Finance",
        "MUTHOOTFIN" : "Muthoot Finance",
        "CHOLAFIN"   : "Cholamandalam Finance",
        "MANAPPURAM" : "Manappuram Finance",
        "SBICARD"    : "SBI Cards",
        "ICICIPRULI" : "ICICI Prudential Life",
        "ICICIGI"    : "ICICI Lombard",
        "HDFCAMC"    : "HDFC AMC",
        "NAUKRI"     : "Info Edge (Naukri)",
        "ZOMATO"     : "Zomato",
        "IRCTC"      : "IRCTC",
        "HAL"        : "Hindustan Aeronautics",
        "BEL"        : "Bharat Electronics",
        "BHEL"       : "Bharat Heavy Electricals",
        "RVNL"       : "Rail Vikas Nigam",
        "IRFC"       : "Indian Railway Finance",
        "PFC"        : "Power Finance Corporation",
        "RECLTD"     : "REC Limited",
        "NHPC"       : "NHPC",
        "SJVN"       : "SJVN",
        "TATAPOWER"  : "Tata Power",
        "TORNTPOWER" : "Torrent Power",
        "GAIL"       : "GAIL India",
        "IGL"        : "Indraprastha Gas",
        "MGL"        : "Mahanagar Gas",
        "PETRONET"   : "Petronet LNG",
        "HINDPETRO"  : "Hindustan Petroleum",
        "IOC"        : "Indian Oil Corporation",
        "ZYDUSLIFE"  : "Zydus Lifesciences",
        "AUROPHARMA" : "Aurobindo Pharma",
        "LUPIN"      : "Lupin",
        "TORNTPHARM" : "Torrent Pharmaceuticals",
        "ALKEM"      : "Alkem Laboratories",
        "IPCALAB"    : "IPCA Laboratories",
        "ABBOTINDIA" : "Abbott India",
        "PFIZER"     : "Pfizer",
        "GLAXO"      : "GlaxoSmithKline Pharma",
        "FORTIS"     : "Fortis Healthcare",
        "MAXHEALTH"  : "Max Healthcare",
        "METROPOLIS" : "Metropolis Healthcare",
        "LALPATHLAB" : "Dr Lal PathLabs",
        "TATAELXSI"  : "Tata Elxsi",
        "LTIM"       : "LTIMindtree",
        "MPHASIS"    : "Mphasis",
        "COFORGE"    : "Coforge",
        "PERSISTENT" : "Persistent Systems",
        "OFSS"       : "Oracle Financial Services",
        "KPIT"       : "KPIT Technologies",
        "CYIENT"     : "Cyient",
        # ── Auto & Manufacturing ───────────────────────────────
        "MOTHERSON"  : "Samvardhana Motherson",
        "BOSCHLTD"   : "Bosch",
        "BALKRISIND" : "Balkrishna Industries",
        "APOLLOTYRE" : "Apollo Tyres",
        "CEATLTD"    : "CEAT",
        "MRF"        : "MRF",
        "ASHOKLEY"   : "Ashok Leyland",
        "TVSMOTOR"   : "TVS Motor",
        "ESCORTS"    : "Escorts Kubota",
        "SONACOMS"   : "Sona BLW Precision",
        "EXIDEIND"   : "Exide Industries",
        "AMARAJABAT" : "Amara Raja Energy",
        "SUNDRMFAST" : "Sundram Fasteners",
        "CRAFTSMAN"  : "Craftsman Automation",
        "SUPRAJIT"   : "Suprajit Engineering",
        # ── Paints & Chemicals ────────────────────────────────
        "BERGEPAINT" : "Berger Paints",
        "PIDILITIND" : "Pidilite Industries",
        "AKZONOBEL"  : "Akzo Nobel India",
        "KANSAINER"  : "Kansai Nerolac Paints",
        "ATUL"       : "Atul Ltd",
        "DEEPAKNTR"  : "Deepak Nitrite",
        "NOCIL"      : "NOCIL",
        "FINEORG"    : "Fine Organic Industries",
        "GALAXYSURF" : "Galaxy Surfactants",
        "NAVINFLUOR" : "Navin Fluorine",
        "SRF"        : "SRF",
        "AARTIIND"   : "Aarti Industries",
        # ── Consumer & FMCG ───────────────────────────────────
        "DABUR"      : "Dabur India",
        "MARICO"     : "Marico",
        "COLPAL"     : "Colgate-Palmolive India",
        "EMAMILTD"   : "Emami",
        "GODREJCP"   : "Godrej Consumer Products",
        "VBL"        : "Varun Beverages",
        "UNITDSPR"   : "United Spirits",
        "UBL"        : "United Breweries",
        "RADICO"     : "Radico Khaitan",
        "GILLETTE"   : "Gillette India",
        "PGHH"       : "Procter & Gamble",
        "GODFRYPHLP" : "Godfrey Phillips",
        "VSTIND"     : "VST Industries",
        "TATACOMM"   : "Tata Communications",
        # ── Retail & Fashion ──────────────────────────────────
        "DMART"      : "Avenue Supermarts",
        "TRENT"      : "Trent",
        "ABFRL"      : "Aditya Birla Fashion",
        "RAYMOND"    : "Raymond",
        "BATAINDIA"  : "Bata India",
        "RELAXO"     : "Relaxo Footwears",
        "PAGEIND"    : "Page Industries",
        "MANYAVAR"   : "Vedant Fashions",
        "SHOPERSTOP" : "Shoppers Stop",
        "VMART"      : "V-Mart Retail",
        # ── Real Estate ───────────────────────────────────────
        "DLF"        : "DLF",
        "GODREJPROP" : "Godrej Properties",
        "OBEROIRLTY" : "Oberoi Realty",
        "PRESTIGE"   : "Prestige Estates",
        "PHOENIXLTD" : "The Phoenix Mills",
        "BRIGADE"    : "Brigade Enterprises",
        "SOBHA"      : "Sobha",
        "LODHA"      : "Macrotech Developers",
        "MAHLIFE"    : "Mahindra Lifespace",
        "KOLTEPATIL" : "Kolte-Patil Developers",
        # ── Metals & Mining ───────────────────────────────────
        "VEDL"       : "Vedanta",
        "NATIONALUM" : "National Aluminium",
        "HINDZINC"   : "Hindustan Zinc",
        "SAIL"       : "Steel Authority of India",
        "JINDALSTEL" : "Jindal Steel & Power",
        "NMDC"       : "NMDC",
        "MOIL"       : "MOIL",
        "RATNAMANI"  : "Ratnamani Metals",
        "WELCORP"    : "Welspun Corp",
        "APL"        : "APL Apollo Tubes",
        # ── Infrastructure & Logistics ────────────────────────
        "GMRINFRA"   : "GMR Airports",
        "CONCOR"     : "Container Corporation",
        "BLUEDART"   : "Blue Dart Express",
        "MAHINDCIE"  : "Mahindra CIE Automotive",
        "ADANITRANS" : "Adani Transmission",
        "AIAENG"     : "AIA Engineering",
        "CUMMINSIND" : "Cummins India",
        "THERMAX"    : "Thermax",
        "BHARTIENE"  : "Bharti Enterprises",
        "IRB"        : "IRB Infrastructure",
        # ── Telecom & Tech ────────────────────────────────────
        "IDEA"       : "Vodafone Idea",
        "HFCL"       : "HFCL",
        "TEJASNET"   : "Tejas Networks",
        "RAILTEL"    : "RailTel Corporation",
        "TANLA"      : "Tanla Platforms",
        "MASTEK"     : "Mastek",
        "ZENSAR"     : "Zensar Technologies",
        "NIITLTD"    : "NIIT",
        "RATEGAIN"   : "RateGain Travel Tech",
        "NYKAA"      : "FSN E-Commerce (Nykaa)",
        "ZENSARTECH" : "Zensar Technologies",
        "DELHIVERY"  : "Delhivery",
        "POLICYBZR"  : "PB Fintech",
        # ── Power & Energy ────────────────────────────────────
        "CESC"       : "CESC",
        "INOXGREEN"  : "INOX Green Energy",
        "GREENPANEL" : "Greenpanel Industries",
        "SUZLON"     : "Suzlon Energy",
        "INOXWIND"   : "Inox Wind",
        "RPOWER"     : "Reliance Power",
        "TORDOC"     : "Torrent Pharma",
        "JPPOWER"    : "Jaiprakash Power",
        # ── Insurance & Fintech ───────────────────────────────
        "NIPPONLIFE" : "Nippon Life India AMC",
        "ABSLAMC"    : "Aditya Birla Sun Life AMC",
        "UTIAMC"     : "UTI AMC",
        "360ONE"     : "360 One WAM",
        "ANGELONE"   : "Angel One",
        "IIFL"       : "IIFL Finance",
        "MOFSL"      : "Motilal Oswal Financial",
        "NUVOCO"     : "Nuvoco Vistas",
        "STARHEALTH" : "Star Health Insurance",
        "GICRE"      : "General Insurance Corp",
        "NIACL"      : "New India Assurance",
        # ── Cement ────────────────────────────────────────────
        "AMBUJACEM"  : "Ambuja Cements",
        "ACC"        : "ACC",
        "DALBHARAT"  : "Dalmia Bharat",
        "JKCEMENT"   : "JK Cement",
        "RAMCOCEM"   : "The Ramco Cements",
        "HEIDELBERG" : "Heidelberg Cement",
        "BIRLACORPN" : "Birla Corporation",
        "SHREECEM"   : "Shree Cement",
        # ── Hospitals & Healthcare ────────────────────────────
        "NARAYANA"   : "Narayana Hrudayalaya",
        "KIMS"       : "KIMS Health",
        "RAINBOW"    : "Rainbow Childrens Medicare",
        "MEDANTA"    : "Global Health (Medanta)",
        "VIJAYA"     : "Vijaya Diagnostic",
        "KRSNAA"     : "Krsnaa Diagnostics",
        # ── Agri & Fertilizers ────────────────────────────────
        "COROMANDEL" : "Coromandel International",
        "CHAMBLFERT" : "Chambal Fertilisers",
        "GNFC"       : "Gujarat Narmada Valley Fert",
        "GSFC"       : "Gujarat State Fertilizers",
        "PIIND"      : "PI Industries",
        "RALLIS"     : "Rallis India",
        "BAYER"      : "Bayer CropScience",
        "DHANUKA"    : "Dhanuka Agritech",
        # ── Electrical & Electronics ──────────────────────────
        "HAVELLS"    : "Havells India",
        "POLYCAB"    : "Polycab India",
        "KEI"        : "KEI Industries",
        "FINOLEX"    : "Finolex Cables",
        "DIXON"      : "Dixon Technologies",
        "AMBER"      : "Amber Enterprises",
        "VOLTAS"     : "Voltas",
        "BLUESTARCO" : "Blue Star",
        "WHIRLPOOL"  : "Whirlpool India",
        "SYMPHONY"   : "Symphony",
        "CROMPTON"   : "Crompton Greaves Consumer",
        "ORIENTELEC" : "Orient Electric",
        "VGUARD"     : "V-Guard Industries",
        "BAJAJCON"   : "Bajaj Consumer Care",
        # ── Defence ───────────────────────────────────────────
        "BEML"       : "BEML",
        "MAZDOCK"    : "Mazagon Dock Shipbuilders",
        "COCHINSHIP" : "Cochin Shipyard",
        "GRSE"       : "Garden Reach Shipbuilders",
        "MIDHANI"    : "Mishra Dhatu Nigam",
        "PARAS"      : "Paras Defence",
        "DATAPATTNS" : "Data Patterns",
        # ── Textiles ──────────────────────────────────────────
        "ARVIND"     : "Arvind",
        "WELSPUNIND" : "Welspun India",
        "TRIDENT"    : "Trident",
        "VARDHMAN"   : "Vardhman Textiles",
        "KITEX"      : "Kitex Garments",
        "GOKEX"      : "Gokaldas Exports",
        # ── Hotels & Hospitality ──────────────────────────────
        "INDHOTEL"   : "Indian Hotels (Taj)",
        "EIHOTEL"    : "EIH (Oberoi Hotels)",
        "LEMONTREE"  : "Lemon Tree Hotels",
        "CHALET"     : "Chalet Hotels",
        "MAHINDRA"   : "Mahindra Holidays",
        # ── Media & Entertainment ─────────────────────────────
        "ZEEL"       : "Zee Entertainment",
        "SUNTV"      : "Sun TV Network",
        "PVRINOX"    : "PVR INOX",
        "SAREGAMA"   : "Saregama India",
        "TIPS"       : "Tips Music",
        # ── Miscellaneous Large Caps ──────────────────────────
        "SIEMENS"    : "Siemens India",
        "ABB"        : "ABB India",
        "HONAUT"     : "Honeywell Automation",
        "3MINDIA"    : "3M India",
        "SCHAEFFLER" : "Schaeffler India",
        "SKFINDIA"   : "SKF India",
        "TIMKEN"     : "Timken India",
        "ASTRAL"     : "Astral",
        "SUPREMEIND" : "Supreme Industries",
        "AARTISURF"  : "Aarti Surfactants",
        "CLEAN"      : "Clean Science",
        "EPIGRAL"    : "Epigral",
        "ROSSARI"    : "Rossari Biotech",
        "TATACHEM"   : "Tata Chemicals",
        "GHCL"       : "GHCL",
        "PCBL"       : "PCBL",
        "GUJGASLTD"  : "Gujarat Gas",
        "ATGL"       : "Adani Total Gas",
        "AEGISLOG"   : "Aegis Logistics",
        "CASTROLIND" : "Castrol India",
        "GULFOILLUB" : "Gulf Oil Lubricants",
        "GOODYEAR"   : "Goodyear India",
        "NRBBEARING" : "NRB Bearing",
        "GRAPHITE"   : "Graphite India",
        "HEG"        : "HEG",
        "TDPOWERSYS" : "TD Power Systems",
        "KEC"        : "KEC International",
        "KALPATPOWR" : "Kalpataru Power",
        "VOLTAMP"    : "Voltamp Transformers",
        "ISGEC"      : "ISGEC Heavy Engineering",
        "ELECON"     : "Elecon Engineering",
        "GRINDWELL"  : "Grindwell Norton",
        "CARBORUNIV" : "Carborundum Universal",
        "CENTURYPLY" : "Century Plyboards",
        "CENTURYTEX" : "Century Textiles",
        "JKPAPER"    : "JK Paper",
        "TNPL"       : "Tamil Nadu Newsprint",
        "WESTLIFE"   : "Westlife Foodworld (McD)",
        "JUBLFOOD"   : "Jubilant Foodworks (Dominos)",
        "DEVYANI"    : "Devyani International (KFC)",
        "SAPPHIRE"   : "Sapphire Foods (KFC)",
        "BIKAJI"     : "Bikaji Foods",
        "BECTORFOOD" : "Mrs. Bectors Food",
        "SULA"       : "Sula Vineyards",
        "GLOBUSSPR"  : "Globus Spirits",
        "TITAGARH"   : "Titagarh Rail Systems",
        "TEXRAIL"    : "Texmaco Rail & Engineering",
        "IRCON"      : "IRCON International",
        "NBCC"       : "NBCC India",
        "HUDCO"      : "HUDCO",
        "CANFINHOME" : "Can Fin Homes",
        "APTUS"      : "Aptus Value Housing Finance",
        "HOMEFIRST"  : "Home First Finance",
        "AAVAS"      : "Aavas Financiers",
        "REPCO"      : "Repco Home Finance",
        "CREDITACC"  : "CreditAccess Grameen",
        "UJJIVANSFB" : "Ujjivan Small Finance Bank",
        "EQUITASBNK" : "Equitas Small Finance Bank",
        "SURYODAY"   : "Suryoday Small Finance Bank",
        "ESAFSFB"    : "ESAF Small Finance Bank",
        "FINPIPE"    : "Finolex Industries",
        "PRINCEPIPE" : "Prince Pipes",
        "APOLLOPIPE" : "Apollo Pipes",
        "SKIPPER"    : "Skipper",
        "MAHSEAMLES" : "Maharashtra Seamless",
        "JINDALSAW"  : "Jindal SAW",
        "WELSPUNLIV" : "Welspun Living",
        "SPORTKING"  : "Sportking India",
        "NSLNISP"    : "NMDC Steel",
    }

    result = {}
    for sym, name in stocks.items():
        result[sym] = {
            'symbol'    : sym,
            'yf_symbol' : f"{sym}.NS",
            'name'      : name
        }
    return result


if __name__ == "__main__":
    stock_map = get_all_nse_stocks()

    print(f"\nTotal stocks fetched: {len(stock_map)}")

    # Save
    with open('nse_stocks.pkl', 'wb') as f:
        pickle.dump(stock_map, f)

    # Save readable CSV too
    rows = [{'Symbol': v['symbol'], 'YF Symbol': v['yf_symbol'], 'Company': v['name']}
            for v in stock_map.values()]
    pd.DataFrame(rows).to_csv('nse_stocks.csv', index=False)

    print(f"Saved to nse_stocks.pkl and nse_stocks.csv")
    print(f"\nSample stocks:")
    for i, (sym, info) in enumerate(list(stock_map.items())[:10]):
        print(f"  {info['yf_symbol']:20} {info['name']}")

    print(f"\nNow run: python train.py")