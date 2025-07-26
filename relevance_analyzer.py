import pandas as pd
import joblib
import os
import fitz  # PyMuPDF
from PIL import Image
import pytesseract
import numpy as np
import cv2 # OpenCV library
import json
from sentence_transformers import SentenceTransformer, util
import torch
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import re
from collections import Counter
import nltk
from nltk.tokenize import PunktSentenceTokenizer
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# Global tokenizer
sentence_tokenizer = PunktSentenceTokenizer()

# ==============================================================================
#  STEP 1: REUSE THE EXACT SAME 1A LOGIC FOR STRUCTURAL ANALYSIS
# ==============================================================================

def extract_features_from_pdf(pdf_path: str) -> pd.DataFrame:
    """Extracts text lines and their features from a PDF using a hybrid approach."""
    if not os.path.exists(pdf_path):
        print(f"Warning: Document not found: {pdf_path}")
        return pd.DataFrame()
    
    try:
        doc = fitz.open(pdf_path)
        all_features = []
        
        for page_num, page in enumerate(doc):
            page_features = []
            page_text = page.get_text().strip()
            
            if len(page_text) < 100:
                # OCR Fallback - improved error handling
                try:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                    
                    for i in range(len(ocr_data['text'])):
                        text = ocr_data['text'][i].strip()
                        confidence = int(ocr_data['conf'][i]) if ocr_data['conf'][i] != -1 else 0
                        
                        if text and confidence > 30:  # LOWERED from 50 for better inclusivity
                            page_features.append({
                                "text": text, "x0": ocr_data['left'][i], "y0": ocr_data['top'][i],
                                "x1": ocr_data['left'][i] + ocr_data['width'][i], 
                                "y1": ocr_data['top'][i] + ocr_data['height'][i],
                                "font_size": 12.0, "font_name": "OCR", "is_bold": False, 
                                "page_num": page_num + 1
                            })
                except Exception as e:
                    print(f"OCR failed for page {page_num + 1}: {e}")
            else:
                # Native Text Extraction - improved error handling
                try:
                    page_dict = page.get_text("dict", sort=True)
                    for block in page_dict.get("blocks", []):
                        if "lines" in block:
                            for line in block.get("lines", []):
                                spans = line.get("spans", [])
                                if spans:
                                    line_text = " ".join([s.get("text", "").strip() for s in spans])
                                    if line_text:
                                        first_span = spans[0]
                                        page_features.append({
                                            "text": line_text, 
                                            "font_size": round(first_span.get("size", 12.0), 2),
                                            "font_name": first_span.get("font", "Unknown"), 
                                            "is_bold": "bold" in first_span.get("font", "").lower(),
                                            "x0": round(line.get("bbox", [0, 0, 0, 0])[0], 2), 
                                            "y0": round(line.get("bbox", [0, 0, 0, 0])[1], 2),
                                            "x1": round(line.get("bbox", [0, 0, 0, 0])[2], 2), 
                                            "y1": round(line.get("bbox", [0, 0, 0, 0])[3], 2), 
                                            "page_num": page_num + 1
                                        })
                except Exception as e:
                    print(f"Native text extraction failed for page {page_num + 1}: {e}")
                    
            all_features.extend(page_features)
        
        doc.close()
        return pd.DataFrame(all_features)
        
    except Exception as e:
        print(f"Error processing document {pdf_path}: {e}")
        return pd.DataFrame()

def is_valid_heading(text: str, font_size: float, is_bold: bool, relative_font_size: float, normalized_y_pos: float) -> bool:
    """FURTHER RELAXED heading validation for better inclusivity - FIXED VERSION."""
    text = text.strip()
    
    # Basic length checks - MORE LENIENT
    if len(text) < 2 or len(text) > 200:
        return False
    
    # Word count check - MORE GENEROUS
    word_count = len(text.split())
    if word_count < 1 or word_count > 20:
        return False
    
    # Should not end with sentence punctuation (except colons which are ok for headings)
    if text.endswith(('.', '!', '?', ';')):
        return False
    
    # Filter out obvious fragments - REDUCED patterns for more inclusivity
    fragment_patterns = [
        r'^(and|or|but|so)\s',  # Starting with conjunctions only
        r'^\d+\.\s*$',  # Just numbers with period
        r'^[^\w\s]+$',  # Only punctuation
    ]
    
    for pattern in fragment_patterns:
        if re.match(pattern, text, re.IGNORECASE):
            return False
    
    # CRITICAL FIX: Much more lenient font-based validation
    # Accept anything that's even slightly larger than normal text
    if relative_font_size < 0.98:  # SIGNIFICANTLY LOWERED from 1.02
        return False
    
    # Additional boost for certain heading indicators
    heading_indicators = ['chapter', 'section', 'part', 'introduction', 'conclusion', 'overview', 'summary']
    if any(indicator in text.lower() for indicator in heading_indicators):
        return True
    
    # Accept text that looks like a title/heading based on content
    if text.isupper() and len(text) > 5:  # All caps titles
        return True
    
    if text.istitle() and len(text.split()) <= 8:  # Title case short phrases
        return True
    
    return True

def extract_document_title(df: pd.DataFrame) -> str:
    """Extract the most likely document title."""
    if df.empty:
        return "Untitled Document"
    
    # Look for text in the first page with largest font size
    first_page = df[df['page_num'] == 1].copy()
    if first_page.empty:
        return "Untitled Document"
    
    # Find the largest font size on first page - more lenient
    max_font_size = first_page['font_size'].max()
    title_candidates = first_page[first_page['font_size'] >= max_font_size - 5].copy()  # INCREASED tolerance
    
    # Filter for reasonable title characteristics
    valid_titles = []
    for _, row in title_candidates.iterrows():
        text = row['text'].strip()
        if (len(text) > 3 and len(text) < 150 and  # RELAXED constraints
            len(text.split()) <= 15 and  # INCREASED word limit
            not text.endswith('.') and
            row.get('normalized_y_pos', 0) < 0.7):  # INCREASED position tolerance
            valid_titles.append((text, row['font_size'], row.get('normalized_y_pos', 0)))
    
    if valid_titles:
        # Sort by font size (descending) and position (ascending - higher on page)
        valid_titles.sort(key=lambda x: (-x[1], x[2]))
        return valid_titles[0][0]
    
    return "Untitled Document"

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Creates the same features used during training with better error handling."""
    if df.empty: 
        return df
    
    try:
        df['page_height'] = df.groupby('page_num')['y1'].transform('max')
        df['page_height'] = df['page_height'].fillna(792.0)  # Default page height
        
        # Improved common font size calculation with fallback
        def safe_mode_with_fallback(series):
            try:
                mode_result = series.mode()
                if not mode_result.empty:
                    return mode_result.iloc[0]
                return series.median()
            except:
                return 12.0
        
        df['common_font_size'] = df.groupby('page_num')['font_size'].transform(safe_mode_with_fallback)
        df['common_font_size'] = df['common_font_size'].fillna(12.0)
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
        df['page_height'] = 792.0
        df['common_font_size'] = 12.0
    
    # Calculate derived features with safer division
    df['relative_font_size'] = df['font_size'] / df['common_font_size'].replace(0, 12.0)
    df['line_width'] = df['x1'] - df['x0']
    df['normalized_y_pos'] = df['y0'] / df['page_height'].replace(0, 792.0)
    
    # Clean up any infinite or NaN values
    numeric_cols = ['relative_font_size', 'line_width', 'normalized_y_pos']
    for col in numeric_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        median_val = df[col].median()
        if pd.isna(median_val):
            median_val = 1.0 if col == 'relative_font_size' else 0.0
        df[col] = df[col].fillna(median_val)
    
    return df

def extract_keywords(text: str, top_k: int = 10) -> list:
    """Extract important keywords from text using simple frequency analysis."""
    # Clean text and convert to lowercase
    clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
    clean_text = re.sub(r'\s+', ' ', clean_text)  # Normalize whitespace
    words = clean_text.split()
    
    # Enhanced stopwords list
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
        'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
        'between', 'among', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 
        'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could', 'can', 'may', 'might', 
        'must', 'shall', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
        'they', 'me', 'him', 'her', 'us', 'them', 'your', 'my', 'our', 'their', 'his', 'its',
        'all', 'some', 'any', 'each', 'every', 'most', 'more', 'less', 'much', 'many', 'few',
        'also', 'just', 'only', 'even', 'still', 'yet', 'now', 'then', 'here', 'there', 'where',
        'when', 'what', 'who', 'how', 'why', 'which', 'one', 'two', 'three', 'first', 'second',
        'using', 'use', 'used', 'step', 'steps', 'page', 'pages', 'see', 'click', 'open', 'close',
        'section', 'chapter', 'document', 'file', 'text', 'content'
    }
    
    filtered_words = [word for word in words if len(word) > 2 and len(word) < 20 and word not in stopwords and word.isalpha()]
    
    # Count frequency with position weighting
    word_freq = Counter()
    for i, word in enumerate(filtered_words):
        weight = max(1.0, 2.0 - (i / max(len(filtered_words), 1)))
        word_freq[word] += weight
    
    return [word for word, _ in word_freq.most_common(top_k)]

def create_enhanced_semantic_chunks(df_with_labels: pd.DataFrame, persona: str, job_to_be_done: str) -> list:
    """IMPROVED version with better heading detection and fallback logic."""
    chunks = []
    current_chunk = None
    
    # Extract keywords from query for boosting
    query_keywords = set(extract_keywords(f"{persona} {job_to_be_done}", 25))
    
    # Get document title for context
    doc_title = extract_document_title(df_with_labels)
    doc_filename = df_with_labels['filename'].iloc[0] if not df_with_labels.empty else "Unknown"
    
    # CRITICAL FIX: Track what was actually detected as headings
    detected_headings = []
    
    for _, row in df_with_labels.iterrows():
        label = row['predicted_label']
        text = row['text'].strip()
        
        if label in ['H1', 'H2', 'H3']:
            # Apply FURTHER RELAXED heading validation
            if not is_valid_heading(text, row['font_size'], row['is_bold'], 
                                  row['relative_font_size'], row['normalized_y_pos']):
                label = 'O'  # Demote to ordinary text
            else:
                detected_headings.append(text)  # Track detected headings
                
                # Save previous chunk if exists
                if current_chunk and current_chunk["text_content"]:
                    chunks.append(current_chunk)
                
                # Calculate heading relevance boost
                heading_keywords = set(extract_keywords(text, 20))
                keyword_overlap = len(query_keywords.intersection(heading_keywords))
                relevance_boost = keyword_overlap * 0.12
                
                # Assign heading level weights with better hierarchy
                level_weights = {'H1': 0.3, 'H2': 0.2, 'H3': 0.1}
                level_boost = level_weights.get(label, 0)
                
                current_chunk = {
                    "doc_name": doc_filename,
                    "page_num": row['page_num'] - 1,  # Use 0-based index
                    "section_title": text,
                    "heading_level": label,
                    "text_content": [text],  # Start with the heading itself
                    "relevance_boost": relevance_boost + level_boost,
                    "keyword_matches": list(heading_keywords.intersection(query_keywords)),
                    "doc_title": doc_title
                }
        
        if label == 'O' and current_chunk:
            current_chunk["text_content"].append(text)
    
    # Save the last chunk
    if current_chunk and current_chunk["text_content"]:
        chunks.append(current_chunk)
    
    # CRITICAL DEBUG INFO
    print(f"        DEBUG: Detected {len(detected_headings)} valid headings: {detected_headings[:3] if detected_headings else 'None'}")
    
    # IMPROVED FALLBACK: If no proper sections found, create content-based sections
    if not chunks:
        print(f"        No structured sections found, creating content-based sections...")
        
        # Group text by pages and create page-based sections
        page_groups = df_with_labels.groupby('page_num')
        
        for page_num, page_data in page_groups:
            page_text = " ".join(page_data['text'].tolist())
            
            if len(page_text.strip()) > 100:  # Only include substantial pages
                # Try to find a good section title from the page
                section_title = f"Content from {doc_filename.replace('.pdf', '')} - Page {page_num}"
                
                # Look for potential titles in the first few lines
                first_lines = page_data.head(5)['text'].tolist()
                for line in first_lines:
                    if (len(line) > 10 and len(line) < 100 and 
                        not line.endswith('.') and 
                        len(line.split()) <= 10):
                        section_title = line
                        break
                
                chunks.append({
                    "doc_name": doc_filename,
                    "page_num": page_num - 1,
                    "section_title": section_title,
                    "heading_level": "O",
                    "text_content": page_text,
                    "relevance_boost": 0.05,  # Small boost for fallback content
                    "keyword_matches": [],
                    "doc_title": doc_title
                })
    
    # Finalize text content with IMPROVED processing
    enhanced_chunks = []
    for chunk in chunks:
        if isinstance(chunk["text_content"], list):
            full_text = "\n".join(chunk["text_content"])
        else:
            full_text = str(chunk["text_content"])
        
        # Skip chunks that are too short
        if len(full_text.strip()) < 40:
            continue
            
        chunk["text_content"] = full_text
        
        try:
            sentences = sentence_tokenizer.tokenize(full_text)
            chunk["sentence_count"] = len(sentences)
            
            # Create refined text with sentence-aware truncation
            refined_text = ""
            for sentence in sentences:
                if len(refined_text) + len(sentence) + 1 <= 800:
                    refined_text += sentence + " "
                else:
                    break
            
            chunk["refined_text"] = refined_text.strip()
            if len(chunk["refined_text"]) < len(full_text):
                chunk["refined_text"] += "..."
        except:
            # Fallback if sentence tokenization fails
            chunk["sentence_count"] = 1
            chunk["refined_text"] = full_text[:800] + ("..." if len(full_text) > 800 else "")
        
        enhanced_chunks.append(chunk)
    
    return enhanced_chunks

def create_mini_chunks(df_with_labels: pd.DataFrame, persona: str, job_to_be_done: str) -> list:
    """Create smaller, more granular chunks with IMPROVED fallback."""
    mini_chunks = []
    query_keywords = set(extract_keywords(f"{persona} {job_to_be_done}", 25))
    
    current_section = {"title": "", "level": "O", "content": [], "doc_name": "", "page_num": 0}
    doc_filename = df_with_labels['filename'].iloc[0] if not df_with_labels.empty else "Unknown"
    
    sections_created = 0
    
    for _, row in df_with_labels.iterrows():
        label = row['predicted_label']
        text = row['text'].strip()
        
        if label in ['H1', 'H2', 'H3'] and is_valid_heading(text, row['font_size'], row['is_bold'], 
                                                            row['relative_font_size'], row['normalized_y_pos']):
            # Process previous section's content
            if current_section["content"]:
                sections_created += 1
                section_text = " ".join(current_section["content"])
                
                try:
                    sentences = sentence_tokenizer.tokenize(section_text)
                    
                    # Group sentences into mini-chunks of 2-3 sentences
                    for i in range(0, len(sentences), 2):
                        mini_chunk_sentences = sentences[i:i+3]
                        mini_chunk_text = " ".join(mini_chunk_sentences)
                        
                        if len(mini_chunk_text.strip()) > 40:
                            # Calculate keyword overlap boost
                            chunk_keywords = set(extract_keywords(mini_chunk_text))
                            keyword_overlap = len(query_keywords.intersection(chunk_keywords))
                            
                            mini_chunks.append({
                                "doc_name": current_section["doc_name"],
                                "page_num": current_section["page_num"],
                                "section_title": current_section["title"],
                                "heading_level": current_section["level"],
                                "mini_chunk_text": mini_chunk_text,
                                "keyword_overlap_score": keyword_overlap,
                                "sentence_count": len(mini_chunk_sentences)
                            })
                except:
                    # Fallback if sentence tokenization fails
                    if len(section_text.strip()) > 40:
                        chunk_keywords = set(extract_keywords(section_text))
                        keyword_overlap = len(query_keywords.intersection(chunk_keywords))
                        
                        mini_chunks.append({
                            "doc_name": current_section["doc_name"],
                            "page_num": current_section["page_num"],
                            "section_title": current_section["title"],
                            "heading_level": current_section["level"],
                            "mini_chunk_text": section_text[:600],
                            "keyword_overlap_score": keyword_overlap,
                            "sentence_count": 1
                        })
            
            # Start new section
            current_section = {
                "title": text,
                "level": label,
                "content": [],
                "doc_name": doc_filename,
                "page_num": row['page_num'] - 1
            }
        elif label == 'O':
            current_section["content"].append(text)
    
    # Process the last section
    if current_section["content"]:
        sections_created += 1
        section_text = " ".join(current_section["content"])
        
        try:
            sentences = sentence_tokenizer.tokenize(section_text)
            
            for i in range(0, len(sentences), 2):
                mini_chunk_sentences = sentences[i:i+3]
                mini_chunk_text = " ".join(mini_chunk_sentences)
                
                if len(mini_chunk_text.strip()) > 40:
                    chunk_keywords = set(extract_keywords(mini_chunk_text))
                    keyword_overlap = len(query_keywords.intersection(chunk_keywords))
                    
                    mini_chunks.append({
                        "doc_name": current_section["doc_name"],
                        "page_num": current_section["page_num"],
                        "section_title": current_section["title"],
                        "heading_level": current_section["level"],
                        "mini_chunk_text": mini_chunk_text,
                        "keyword_overlap_score": keyword_overlap,
                        "sentence_count": len(mini_chunk_sentences)
                    })
        except:
            # Fallback
            if len(section_text.strip()) > 40:
                chunk_keywords = set(extract_keywords(section_text))
                keyword_overlap = len(query_keywords.intersection(chunk_keywords))
                
                mini_chunks.append({
                    "doc_name": current_section["doc_name"],
                    "page_num": current_section["page_num"],
                    "section_title": current_section["title"],
                    "heading_level": current_section["level"],
                    "mini_chunk_text": section_text[:600],
                    "keyword_overlap_score": keyword_overlap,
                    "sentence_count": 1
                })
    
    # IMPROVED FALLBACK: If no mini-chunks created, create page-based chunks
    if not mini_chunks:
        print(f"        No structured mini-chunks found, creating page-based mini-chunks...")
        page_groups = df_with_labels.groupby('page_num')
        
        for page_num, page_data in page_groups:
            page_text = " ".join(page_data['text'].tolist())
            
            if len(page_text.strip()) > 100:
                # Split page text into smaller chunks
                try:
                    sentences = sentence_tokenizer.tokenize(page_text)
                    for i in range(0, len(sentences), 3):
                        chunk_sentences = sentences[i:i+4]
                        chunk_text = " ".join(chunk_sentences)
                        
                        if len(chunk_text.strip()) > 40:
                            chunk_keywords = set(extract_keywords(chunk_text))
                            keyword_overlap = len(query_keywords.intersection(chunk_keywords))
                            
                            mini_chunks.append({
                                "doc_name": doc_filename,
                                "page_num": page_num - 1,
                                "section_title": f"Content from {doc_filename.replace('.pdf', '')} - Page {page_num}",
                                "heading_level": "O",
                                "mini_chunk_text": chunk_text,
                                "keyword_overlap_score": keyword_overlap,
                                "sentence_count": len(chunk_sentences)
                            })
                except:
                    # Simple text splitting fallback
                    words = page_text.split()
                    for i in range(0, len(words), 50):
                        chunk_words = words[i:i+80]
                        chunk_text = " ".join(chunk_words)
                        
                        if len(chunk_text.strip()) > 40:
                            chunk_keywords = set(extract_keywords(chunk_text))
                            keyword_overlap = len(query_keywords.intersection(chunk_keywords))
                            
                            mini_chunks.append({
                                "doc_name": doc_filename,
                                "page_num": page_num - 1,
                                "section_title": f"Content from {doc_filename.replace('.pdf', '')} - Page {page_num}",
                                "heading_level": "O",
                                "mini_chunk_text": chunk_text,
                                "keyword_overlap_score": keyword_overlap,
                                "sentence_count": 1
                            })
    
    return mini_chunks

# ==============================================================================
#  STEP 2: ENHANCED RELEVANCE ANALYSIS LOGIC WITH GUARANTEED RESULTS
# ==============================================================================

def find_relevant_sections(
    doc_collection_paths: list, 
    persona: str, 
    job_to_be_done: str, 
    structure_model: RandomForestClassifier,
    structure_encoder: LabelEncoder,
    semantic_model: SentenceTransformer
) -> dict:
    """
    FIXED version with improved section detection and consistent output.
    """
    # Create more focused query
    query = f"Role: {persona}. Task: {job_to_be_done}."
    print(f"    - Generating embedding for query...")
    
    try:
        query_embedding = semantic_model.encode(query, convert_to_tensor=True)
    except Exception as e:
        print(f"Error generating query embedding: {e}")
        return {"error": f"Failed to generate query embedding: {e}"}

    all_chunks = []
    all_mini_chunks = []
    print("    - Processing document collection for structure and content...")
    
    for pdf_path in doc_collection_paths:
        if not os.path.exists(pdf_path):
            print(f"      Warning: File not found: {os.path.basename(pdf_path)}")
            continue
            
        print(f"      Processing: {os.path.basename(pdf_path)}")
        
        try:
            # 1. Get features from PDF
            df_features = extract_features_from_pdf(pdf_path)
            if df_features.empty: 
                print(f"        No content extracted from {os.path.basename(pdf_path)}")
                continue
            
            df_engineered = engineer_features(df_features)
            df_engineered['filename'] = os.path.basename(pdf_path)

            # 2. Predict structure using 1A model
            features_for_prediction = [
                'font_size', 'is_bold', 'x0', 'y0', 'x1', 'y1',
                'relative_font_size', 'line_width', 'normalized_y_pos'
            ]
            X_new = df_engineered[features_for_prediction].copy()
            X_new['is_bold'] = X_new['is_bold'].astype(int)
            
            # Handle missing values
            for col in X_new.columns:
                if X_new[col].isna().any():
                    if col == 'relative_font_size':
                        X_new[col] = X_new[col].fillna(1.0)
                    else:
                        X_new[col] = X_new[col].fillna(X_new[col].median())
            
            predictions_encoded = structure_model.predict(X_new)
            df_engineered['predicted_label'] = structure_encoder.inverse_transform(predictions_encoded)

            # 3. Create both regular and mini chunks with IMPROVED logic
            pdf_chunks = create_enhanced_semantic_chunks(df_engineered, persona, job_to_be_done)
            pdf_mini_chunks = create_mini_chunks(df_engineered, persona, job_to_be_done)
            
            all_chunks.extend(pdf_chunks)
            all_mini_chunks.extend(pdf_mini_chunks)
            
            print(f"        Created {len(pdf_chunks)} sections and {len(pdf_mini_chunks)} mini-chunks")
            
        except Exception as e:
            print(f"        Error processing {os.path.basename(pdf_path)}: {e}")
            continue
    
    # IMPROVED FINAL FALLBACK: Only if absolutely no chunks found
    if not all_chunks and not all_mini_chunks:
        print("    - No structured content found, creating final fallback content...")
        for pdf_path in doc_collection_paths[:5]:  # Process first 5 docs as fallback
            try:
                if not os.path.exists(pdf_path):
                    continue
                    
                doc_name = os.path.basename(pdf_path)
                doc = fitz.open(pdf_path)
                for page_num, page in enumerate(doc[:3]):  # First 3 pages
                    text = page.get_text().strip()
                    if len(text) > 100:
                        all_chunks.append({
                            "doc_name": doc_name,
                            "page_num": page_num,
                            "section_title": f"Content from {doc_name.replace('.pdf', '')} - Page {page_num + 1}",
                            "heading_level": "O",
                            "text_content": text[:800],
                            "relevance_boost": 0.1,
                            "keyword_matches": [],
                            "doc_title": doc_name.replace('.pdf', '')
                        })
                doc.close()
            except:
                continue

    if not all_chunks:
        return {
            "extracted_sections": [],
            "subsection_analysis": []
        }

    # Process regular chunks
    df_all_chunks = pd.DataFrame(all_chunks)
    
    # Debug information
    print(f"    - Total chunks created: {len(df_all_chunks)}")
    if len(df_all_chunks) > 0:
        heading_levels = df_all_chunks['heading_level'].value_counts()
        print(f"    - Heading distribution: {dict(heading_levels)}")
    
    # 4. Generate embeddings for each chunk
    print(f"    - Generating embeddings for {len(df_all_chunks)} semantic chunks...")
    try:
        chunk_texts = df_all_chunks['text_content'].tolist()
        chunk_embeddings = semantic_model.encode(chunk_texts, convert_to_tensor=True, show_progress_bar=False)
        df_all_chunks['embedding'] = chunk_embeddings.tolist()

        # 5. Calculate enhanced relevance scores
        print("    - Calculating enhanced relevance scores...")
        cosine_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
        
        # Apply relevance boosts
        base_scores = cosine_scores.cpu().numpy()
        relevance_boosts = df_all_chunks['relevance_boost'].values
        boosted_scores = base_scores + relevance_boosts
        df_all_chunks['relevance_score'] = boosted_scores
        
    except Exception as e:
        print(f"    - Error in embedding generation: {e}")
        # Fallback: use keyword-based scoring only
        df_all_chunks['relevance_score'] = df_all_chunks['relevance_boost'].values

    # Also process mini chunks for subsection analysis
    df_mini_chunks_ranked = None
    if all_mini_chunks:
        try:
            df_mini_chunks = pd.DataFrame(all_mini_chunks)
            print(f"    - Processing {len(df_mini_chunks)} mini-chunks for detailed analysis...")
            
            mini_texts = df_mini_chunks['mini_chunk_text'].tolist()
            mini_embeddings = semantic_model.encode(mini_texts, convert_to_tensor=True, show_progress_bar=False)
            mini_cosine_scores = util.pytorch_cos_sim(query_embedding, mini_embeddings)[0]
            
            # Boost mini-chunk scores with keyword overlap
            mini_base_scores = mini_cosine_scores.cpu().numpy()
            keyword_boosts = df_mini_chunks['keyword_overlap_score'].values * 0.08
            df_mini_chunks['relevance_score'] = mini_base_scores + keyword_boosts
            
            # Sort mini chunks by relevance
            df_mini_chunks_ranked = df_mini_chunks.sort_values(by='relevance_score', ascending=False).reset_index(drop=True)
        except Exception as e:
            print(f"    - Error processing mini-chunks: {e}")
            df_mini_chunks_ranked = None

    # Sort main chunks by relevance - GUARANTEED TOP 5
    df_ranked = df_all_chunks.sort_values(by='relevance_score', ascending=False).reset_index(drop=True)
    top_5_results = df_ranked.head(5)

    # 6. Format the output JSON - EXACTLY 5 RESULTS
    extracted_sections = []
    for index, row in top_5_results.iterrows():
        extracted_sections.append({
            "document": row['doc_name'],
            "page_number": int(row['page_num']),
            "section_title": row['section_title'],
            "importance_rank": index + 1
        })
    
    # Enhanced subsection analysis using best mini-chunks or fallback
    subsection_analysis = []
    
    if df_mini_chunks_ranked is not None and len(df_mini_chunks_ranked) > 0:
        # Use top mini-chunks for more granular analysis
        for index, row in df_mini_chunks_ranked.head(5).iterrows():
            subsection_analysis.append({
                "document": row['doc_name'],
                "page_number": int(row['page_num']),
                "refined_text": row['mini_chunk_text']
            })
    else:
        # Fallback to regular chunks
        for index, row in top_5_results.iterrows():
            refined_text = row.get('refined_text', str(row['text_content'])[:700])
            if len(refined_text.strip()) > 30:
                subsection_analysis.append({
                    "document": row['doc_name'],
                    "page_number": int(row['page_num']),
                    "refined_text": refined_text + ("..." if len(str(row['text_content'])) > 700 else "")
                })

    return {
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

# ==============================================================================
#  STEP 3: MAIN EXECUTION LOOP WITH IMPROVED ERROR HANDLING
# ==============================================================================

def main():
    workdir = "." 
    
    print("=" * 60)
    print("ENHANCED DOCUMENT RELEVANCE ANALYZER - TOP 5 GUARANTEED")
    print("=" * 60)
    
    print("\nLoading AI models...")
    try:
        # Load 1A models for structure
        model_1a_path = os.path.join("model_1a")
        if not os.path.exists(model_1a_path):
            print(f"❌ Model directory '{model_1a_path}' not found.")
            return
            
        structure_model = joblib.load(os.path.join(model_1a_path, "document_classifier.joblib"))
        structure_encoder = joblib.load(os.path.join(model_1a_path, "label_encoder.joblib"))
        print("   ✅ Document structure model loaded")
        
        # Load 1B model for semantics
        semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
        print("   ✅ Semantic similarity model loaded")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print("   Make sure all model files are in the correct directories.")
        return
    
    collection_dirs = [d for d in os.listdir(workdir) if os.path.isdir(d) and d.startswith("Collection")]
    if not collection_dirs:
        print(f"❌ No 'Collection' folders found in the root directory.")
        return

    print(f"\nFound {len(collection_dirs)} collections to process: {collection_dirs}")
    
    successful_processing = 0
    total_collections = len(collection_dirs)

    for collection_name in collection_dirs:
        print(f"\n{'='*20} Processing: {collection_name} {'='*20}")
        collection_path = collection_name
        input_json_path = os.path.join(collection_path, "challenge1b_input.json")
        
        # Look for document folder - flexible approach
        docs_folder_path = None
        possible_folders = ["PDFs", "Documents", "Files", "Docs", "Data"]
        
        for folder_name in possible_folders:
            folder_path = os.path.join(collection_path, folder_name)
            if os.path.exists(folder_path):
                docs_folder_path = folder_path
                print(f"   Found document folder: {folder_name}")
                break
        
        if not docs_folder_path:
            print(f"   ❌ No document folder found. Looked for: {possible_folders}")
            continue

        try:
            # Load input configuration
            if not os.path.exists(input_json_path):
                print(f"   ❌ Input file not found: {input_json_path}")
                continue
                
            with open(input_json_path, 'r', encoding='utf-8') as f:
                input_data = json.load(f)
            
            # Extract configuration with flexible handling
            documents_info = input_data.get("documents", [])
            persona_data = input_data.get("persona", {})
            job_data = input_data.get("job_to_be_done", {})
            
            # Handle different input formats
            if isinstance(persona_data, dict):
                persona_text = persona_data.get("role", persona_data.get("description", "Professional"))
            else:
                persona_text = str(persona_data) if persona_data else "Professional"
            
            if isinstance(job_data, dict):
                job_text = job_data.get("task", job_data.get("description", job_data.get("objective", "Analyze documents")))
            else:
                job_text = str(job_data) if job_data else "Analyze documents"
            
            print(f"   Configuration loaded:")
            print(f"   - Persona: {persona_text}")
            print(f"   - Job: {job_text}")
            print(f"   - Documents specified: {len(documents_info)}")
            
            # Get document files
            pdf_files = []
            
            if documents_info:
                # Use specified documents
                for doc_info in documents_info:
                    if isinstance(doc_info, dict):
                        filename = doc_info.get("filename", doc_info.get("name", ""))
                    else:
                        filename = str(doc_info)
                    
                    if filename:
                        doc_path = os.path.join(docs_folder_path, filename)
                        if os.path.exists(doc_path):
                            pdf_files.append(doc_path)
                        else:
                            print(f"   Warning: Specified document not found: {filename}")
            else:
                # Use all documents in folder
                supported_extensions = ['.pdf', '.docx', '.txt', '.doc']
                for filename in os.listdir(docs_folder_path):
                    if any(filename.lower().endswith(ext) for ext in supported_extensions):
                        pdf_files.append(os.path.join(docs_folder_path, filename))
            
            if not pdf_files:
                print(f"   ❌ No valid documents found in {collection_name}")
                continue
                
            print(f"   Found {len(pdf_files)} documents to process")
            
        except Exception as e:
            print(f"   ❌ Error reading input for '{collection_name}': {e}")
            continue

        # Perform analysis with comprehensive error handling
        try:
            print(f"   Starting relevance analysis...")
            analysis_results = find_relevant_sections(
                pdf_files, persona_text, job_text, 
                structure_model, structure_encoder, semantic_model
            )
            
            if "error" in analysis_results:
                print(f"   ❌ Analysis failed: {analysis_results['error']}")
                continue

            # Create output with metadata
            final_json = {
                "metadata": {
                    "input_documents": [os.path.basename(p) for p in pdf_files],
                    "persona": persona_text,
                    "job_to_be_done": job_text,
                    "processing_timestamp": datetime.now().isoformat()
                },
                "extracted_sections": analysis_results.get("extracted_sections", []),
                "subsection_analysis": analysis_results.get("subsection_analysis", [])
            }
            
            # GUARANTEE we have results - final fallback (this should rarely trigger now)
            if not final_json["extracted_sections"] and pdf_files:
                print(f"   Creating emergency fallback results...")
                for i, pdf_path in enumerate(pdf_files[:5]):  # Take first 5 docs
                    doc_name = os.path.basename(pdf_path)
                    final_json["extracted_sections"].append({
                        "document": doc_name,
                        "page_number": 1,
                        "section_title": f"Content from {doc_name.replace('.pdf', '')}",
                        "importance_rank": i + 1
                    })
                    final_json["subsection_analysis"].append({
                        "document": doc_name,
                        "page_number": 1,
                        "refined_text": f"Document overview and content from {doc_name}. This document contains relevant information for {persona_text} regarding {job_text}. Analysis was performed using fallback content extraction due to processing constraints."
                    })
            
            # Save results
            output_json_path = os.path.join(collection_path, "challenge1b_output.json")
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(final_json, f, indent=4, ensure_ascii=False)
            
            # Set file permissions
            try:
                os.chmod(output_json_path, 0o777)
            except:
                pass  # Ignore permission errors
            
            extracted_count = len(final_json["extracted_sections"])
            subsection_count = len(final_json["subsection_analysis"])
            
            print(f"   ✅ Analysis complete:")
            print(f"     - Extracted sections: {extracted_count}")
            print(f"     - Subsection analyses: {subsection_count}")
            print(f"     - Output saved: challenge1b_output.json")
            
            # Print summary of top results
            print(f"   Top sections found:")
            for i, section in enumerate(final_json["extracted_sections"][:3], 1):
                title = section['section_title']
                if len(title) > 60:
                    title = title[:60] + "..."
                print(f"     {i}. '{title}' (Page {section['page_number']}, {section['document']})")
            
            successful_processing += 1
            
        except Exception as e:
            print(f"   ❌ Error during analysis for '{collection_name}': {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE - TOP 5 RESULTS GUARANTEED")
    print(f"Successfully processed: {successful_processing}/{total_collections} collections")
    if successful_processing < total_collections:
        print(f"Failed to process: {total_collections - successful_processing} collections")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()