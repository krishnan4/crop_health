# main.py
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import os
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

from database import init_db, get_db
from auth import hash_password, verify_password, generate_otp, get_otp_expiry, is_otp_valid, send_otp_email
from ai_model import predict_disease

load_dotenv()

app = FastAPI(title="CropHealth AI")

templates = Jinja2Templates(directory="templates")

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.on_event("startup")
async def startup():
    init_db()


def get_current_user(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        return None
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    db.close()
    return user


# ====================== PAGE ROUTES ======================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    user = get_current_user(request)
    return templates.TemplateResponse("index.html", {"request": request, "user": user})


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {
        "request": request, "message": None, "tab": "login", "email": ""
    })


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    db = get_db()
    scans = db.execute(
        "SELECT * FROM crop_scans WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
        (user["id"],)
    ).fetchall()
    db.close()

    return templates.TemplateResponse("dashboard.html", {
        "request": request, "user": user, "scans": scans
    })


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)
    return templates.TemplateResponse("upload.html", {"request": request, "user": user})


@app.get("/feed", response_class=HTMLResponse)
async def feed_page(request: Request):
    user = get_current_user(request)
    db = get_db()

    posts = db.execute("""
        SELECT * FROM feed_posts
        ORDER BY created_at DESC LIMIT 50
    """).fetchall()

    post_list = []
    for post in posts:
        comments = db.execute("""
            SELECT * FROM post_comments
            WHERE post_id = ? ORDER BY created_at ASC
        """, (post["id"],)).fetchall()
        post_list.append({
            "post": dict(post),
            "comments": [dict(c) for c in comments]
        })

    db.close()

    return templates.TemplateResponse("feed.html", {
        "request": request,
        "user": user,
        "posts": post_list
    })


# ====================== AUTH ROUTES ======================

@app.post("/register")
async def register(
    request: Request,
    name: str = Form(...),
    reg_email: str = Form(...),
    reg_password: str = Form(...)
):
    db = get_db()
    existing = db.execute("SELECT id FROM users WHERE email = ?", (reg_email,)).fetchone()
    if existing:
        db.close()
        return templates.TemplateResponse("login.html", {
            "request": request, "message": "❌ Email already registered.", "tab": "register", "email": ""
        })

    hashed = hash_password(reg_password)
    db.execute(
        "INSERT INTO users (name, email, password, is_verified) VALUES (?, ?, ?, 1)",
        (name, reg_email, hashed)
    )
    db.commit()
    db.close()

    return templates.TemplateResponse("login.html", {
        "request": request, "message": "✅ Account created! Please login.", "tab": "login", "email": ""
    })


@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...)):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    db.close()

    if not user or not verify_password(password, user["password"]):
        return templates.TemplateResponse("login.html", {
            "request": request, "message": "❌ Invalid email or password.", "tab": "login", "email": ""
        })

    otp = generate_otp()
    expiry = get_otp_expiry()
    db = get_db()
    db.execute("UPDATE users SET otp = ?, otp_expires = ? WHERE id = ?", (otp, expiry, user["id"]))
    db.commit()
    db.close()

    send_otp_email(email, otp, user["name"])

    return templates.TemplateResponse("login.html", {
        "request": request,
        "message": f"📧 OTP sent to {email}",
        "tab": "otp",
        "email": email
    })


@app.post("/verify-otp")
async def verify_otp(request: Request, email: str = Form(...), otp: str = Form(...)):
    db = get_db()
    user = db.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()

    if not user or user["otp"] != otp or not is_otp_valid(user["otp_expires"]):
        db.close()
        return templates.TemplateResponse("login.html", {
            "request": request, "message": "❌ Invalid or expired OTP.", "tab": "otp", "email": email
        })

    db.execute("UPDATE users SET otp = NULL, otp_expires = NULL WHERE id = ?", (user["id"],))
    db.commit()
    db.close()

    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie("user_id", str(user["id"]), max_age=86400)
    return response


@app.get("/logout")
async def logout():
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("user_id")
    return response


# ====================== SCAN ROUTE ======================

@app.post("/scan", response_class=HTMLResponse)
async def scan_crop(request: Request, image: UploadFile = File(...)):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    file_ext = image.filename.split(".")[-1]
    filename = f"{uuid.uuid4()}.{file_ext}"
    image_path = f"uploads/{filename}"

    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    prediction = predict_disease(image_path)
    treatment = prediction.get("treatment", "Consult a local agronomist.")

    db = get_db()
    db.execute("""
        INSERT INTO crop_scans (user_id, image_path, disease_name, confidence, treatment)
        VALUES (?, ?, ?, ?, ?)
    """, (user["id"], image_path, prediction["disease_name"], prediction["confidence"], treatment))
    db.commit()
    db.close()

    return templates.TemplateResponse("result.html", {
        "request": request,
        "user": user,
        "image_path": f"/{image_path}",
        "disease_name": prediction["disease_name"],
        "confidence": prediction["confidence"],
        "severity": prediction.get("severity", ""),
        "prevention": prediction.get("prevention", ""),
        "all_predictions": prediction.get("all_predictions", []),
        "treatment": treatment
    })


# ====================== FEED ROUTES ======================

@app.post("/feed/post")
async def create_post(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
    disease_tag: str = Form(""),
    image: UploadFile = File(None)
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    image_path = None
    if image and image.filename:
        ext = image.filename.split(".")[-1]
        fname = f"feed_{uuid.uuid4()}.{ext}"
        image_path = f"uploads/{fname}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

    db = get_db()
    db.execute("""
        INSERT INTO feed_posts (user_id, user_name, title, content, image_path, disease_tag)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (user["id"], user["name"], title, content, image_path, disease_tag))
    db.commit()
    db.close()

    return RedirectResponse("/feed", status_code=302)


@app.post("/feed/like/{post_id}")
async def like_post(post_id: int):
    db = get_db()
    db.execute("UPDATE feed_posts SET likes = likes + 1 WHERE id = ?", (post_id,))
    db.commit()
    likes = db.execute("SELECT likes FROM feed_posts WHERE id = ?", (post_id,)).fetchone()["likes"]
    db.close()
    return JSONResponse({"likes": likes})


@app.post("/feed/comment/{post_id}")
async def add_comment(post_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False, "error": "Not authenticated"}, status_code=401)

    try:
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"success": False, "error": "Empty comment"}, status_code=400)

        db = get_db()
        db.execute("""
            INSERT INTO post_comments (post_id, user_id, user_name, content)
            VALUES (?, ?, ?, ?)
        """, (post_id, user["id"], user["name"], content))
        db.commit()

        comment = db.execute("SELECT * FROM post_comments WHERE id = last_insert_rowid()").fetchone()
        db.close()

        return JSONResponse({"success": True, "comment": dict(comment)})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/feed/comment/{comment_id}/edit")
async def edit_comment(comment_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False, "error": "Not authenticated"}, status_code=401)

    try:
        body = await request.json()
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"success": False, "error": "Empty content"}, status_code=400)

        db = get_db()
        db.execute("UPDATE post_comments SET content = ? WHERE id = ? AND user_id = ?",
                   (content, comment_id, user["id"]))
        db.commit()
        db.close()
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/feed/comment/{comment_id}/delete")
async def delete_comment(comment_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False, "error": "Not authenticated"}, status_code=401)

    db = get_db()
    db.execute("DELETE FROM post_comments WHERE id = ? AND user_id = ?", (comment_id, user["id"]))
    db.commit()
    db.close()
    return JSONResponse({"success": True})


@app.get("/feed/comments/{post_id}")
async def get_comments(post_id: int):
    db = get_db()
    comments = db.execute(
        "SELECT * FROM post_comments WHERE post_id = ? ORDER BY created_at ASC",
        (post_id,)
    ).fetchall()
    db.close()
    return JSONResponse([dict(c) for c in comments])


# ====================== PROFILE ROUTES ======================

@app.get("/profile/{user_id}", response_class=HTMLResponse)
async def profile_page(user_id: int, request: Request):
    current_user = get_current_user(request)
    db = get_db()

    profile_user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if not profile_user:
        db.close()
        return HTMLResponse("<h2 style='font-family:sans-serif;padding:3rem'>User not found.</h2>", status_code=404)

    is_owner = current_user and current_user["id"] == user_id

    scans, posts, diseases_found = [], [], 0
    if is_owner:
        scans = [dict(s) for s in db.execute(
            "SELECT * FROM crop_scans WHERE user_id = ? ORDER BY created_at DESC LIMIT 10",
            (user_id,)
        ).fetchall()]
        diseases_found = sum(
            1 for s in scans
            if s["disease_name"] and "healthy" not in s["disease_name"].lower()
        )

    posts = [dict(p) for p in db.execute(
        "SELECT * FROM feed_posts WHERE user_id = ? ORDER BY created_at DESC LIMIT 20",
        (user_id,)
    ).fetchall()]

    followers_count = db.execute(
        "SELECT COUNT(*) as c FROM follows WHERE following_id = ? AND status = 'accepted'",
        (user_id,)
    ).fetchone()["c"]
    following_count = db.execute(
        "SELECT COUNT(*) as c FROM follows WHERE follower_id = ? AND status = 'accepted'",
        (user_id,)
    ).fetchone()["c"]

    follow_status = None
    follows_you = False
    if current_user and current_user["id"] != user_id:
        row = db.execute(
            "SELECT status FROM follows WHERE follower_id = ? AND following_id = ?",
            (current_user["id"], user_id)
        ).fetchone()
        follow_status = row["status"] if row else None

        reverse = db.execute(
            "SELECT status FROM follows WHERE follower_id = ? AND following_id = ?",
            (user_id, current_user["id"])
        ).fetchone()
        follows_you = reverse and reverse["status"] == "accepted"

    follow_requests = []
    if is_owner:
        reqs = db.execute("""
            SELECT f.id, f.follower_id, u.name, u.avatar_path, f.created_at
            FROM follows f JOIN users u ON f.follower_id = u.id
            WHERE f.following_id = ? AND f.status = 'pending'
            ORDER BY f.created_at DESC
        """, (user_id,)).fetchall()
        follow_requests = [dict(r) for r in reqs]

    db.close()

    return templates.TemplateResponse("profile.html", {
        "request": request,
        "user": dict(current_user) if current_user else None,
        "profile_user": dict(profile_user),
        "scans": scans,
        "posts": posts,
        "diseases_found": diseases_found,
        "is_owner": is_owner,
        "followers_count": followers_count,
        "following_count": following_count,
        "follow_status": follow_status,
        "follows_you": follows_you,
        "follow_requests": follow_requests,
    })


@app.post("/profile/edit", response_class=HTMLResponse)
async def edit_profile(
    request: Request,
    name: str = Form(...),
    bio: str = Form(""),
    avatar: UploadFile = File(None)
):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    avatar_path = user["avatar_path"] if user["avatar_path"] else ""
    if avatar and avatar.filename:
        ext = avatar.filename.split(".")[-1].lower()
        fname = f"avatar_{user['id']}_{uuid.uuid4().hex[:8]}.{ext}"
        avatar_path = f"uploads/{fname}"
        with open(avatar_path, "wb") as f:
            shutil.copyfileobj(avatar.file, f)

    db = get_db()
    db.execute(
        "UPDATE users SET name = ?, bio = ?, avatar_path = ? WHERE id = ?",
        (name.strip(), bio.strip(), avatar_path, user["id"])
    )
    db.commit()
    db.close()
    return RedirectResponse(f"/profile/{user['id']}", status_code=302)


# ====================== FOLLOW ROUTES ======================

@app.post("/follow/{target_id}")
async def follow_user(target_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False, "error": "Not authenticated"}, status_code=401)
    if user["id"] == target_id:
        return JSONResponse({"success": False, "error": "Cannot follow yourself"})

    db = get_db()
    existing = db.execute(
        "SELECT * FROM follows WHERE follower_id = ? AND following_id = ?",
        (user["id"], target_id)
    ).fetchone()

    if existing:
        db.close()
        return JSONResponse({"success": False, "error": "Already following or requested"})

    db.execute(
        "INSERT INTO follows (follower_id, following_id, status) VALUES (?, ?, 'accepted')",
        (user["id"], target_id)
    )

    # ── NEW: Create a follow notification for the target user ──
    notif_content = f"{user['name']} started following you."
    db.execute("""
        INSERT INTO notifications (user_id, type, content, related_user_id)
        VALUES (?, 'follow', ?, ?)
    """, (target_id, notif_content, user["id"]))

    db.commit()
    db.close()
    return JSONResponse({"success": True, "status": "accepted"})


@app.post("/unfollow/{target_id}")
async def unfollow_user(target_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False, "error": "Not authenticated"}, status_code=401)

    db = get_db()
    db.execute(
        "DELETE FROM follows WHERE follower_id = ? AND following_id = ?",
        (user["id"], target_id)
    )
    db.commit()
    db.close()
    return JSONResponse({"success": True})


@app.post("/follow/accept/{requester_id}")
async def accept_follow(requester_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False}, status_code=401)

    db = get_db()
    db.execute(
        "UPDATE follows SET status = 'accepted' WHERE follower_id = ? AND following_id = ? AND status = 'pending'",
        (requester_id, user["id"])
    )

    # Notify the requester that their request was accepted
    requester = db.execute("SELECT name FROM users WHERE id = ?", (requester_id,)).fetchone()
    if requester:
        db.execute("""
            INSERT INTO notifications (user_id, type, content, related_user_id)
            VALUES (?, 'follow_accept', ?, ?)
        """, (requester_id, f"{user['name']} accepted your follow request.", user["id"]))

    db.commit()
    db.close()
    return RedirectResponse(f"/profile/{user['id']}", status_code=302)


@app.post("/follow/decline/{requester_id}")
async def decline_follow(requester_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False}, status_code=401)

    db = get_db()
    db.execute(
        "DELETE FROM follows WHERE follower_id = ? AND following_id = ? AND status = 'pending'",
        (requester_id, user["id"])
    )
    db.commit()
    db.close()
    return RedirectResponse(f"/profile/{user['id']}", status_code=302)


# ====================== MESSAGES ROUTES ======================

@app.get("/messages", response_class=HTMLResponse)
async def messages_page(request: Request, to: int = None):
    user = get_current_user(request)
    if not user:
        return RedirectResponse("/login", status_code=302)

    db = get_db()

    # Get all unique conversation partners using a reliable subquery
    partner_rows = db.execute("""
        SELECT DISTINCT other_user_id FROM (
            SELECT receiver_id AS other_user_id FROM messages WHERE sender_id = ?
            UNION
            SELECT sender_id AS other_user_id FROM messages WHERE receiver_id = ?
        )
    """, (user["id"], user["id"])).fetchall()

    conversations = []
    for row in partner_rows:
        other_id = row["other_user_id"]
        other_user = db.execute("SELECT id, name, avatar_path FROM users WHERE id = ?", (other_id,)).fetchone()
        if not other_user:
            continue

        last_msg = db.execute("""
            SELECT * FROM messages
            WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
            ORDER BY created_at DESC LIMIT 1
        """, (user["id"], other_id, other_id, user["id"])).fetchone()

        unread = db.execute("""
            SELECT COUNT(*) as cnt FROM messages
            WHERE sender_id = ? AND receiver_id = ? AND is_read = 0
        """, (other_id, user["id"])).fetchone()["cnt"]

        last_content = ""
        last_time = ""
        if last_msg:
            c = last_msg["content"]
            last_content = (c[:40] + "…") if len(c) > 40 else c
            last_time = last_msg["created_at"]

        conversations.append({
            "other_user_id": other_id,
            "other_name": other_user["name"],
            "other_avatar": other_user["avatar_path"] or "",
            "last_message": last_content,
            "last_time": last_time,
            "unread_count": unread,
        })

    # Sort: unread first, then by most recent message
    conversations.sort(key=lambda c: (-(c["unread_count"] > 0), c["last_time"] or ""), reverse=False)
    conversations.sort(key=lambda c: c["last_time"] or "", reverse=True)
    # Bring unread convos to the top within recency sort
    conversations.sort(key=lambda c: (0 if c["unread_count"] > 0 else 1, c["last_time"] or ""), reverse=False)

    # If opening a new conversation not yet in the list, prepend it
    if to:
        existing_ids = [c["other_user_id"] for c in conversations]
        if to not in existing_ids:
            to_user = db.execute("SELECT id, name, avatar_path FROM users WHERE id = ?", (to,)).fetchone()
            if to_user:
                conversations.insert(0, {
                    "other_user_id": to,
                    "other_name": to_user["name"],
                    "other_avatar": to_user["avatar_path"] or "",
                    "last_message": "",
                    "last_time": "",
                    "unread_count": 0,
                })

    active_user = None
    messages = []
    if to:
        active_user_row = db.execute("SELECT * FROM users WHERE id = ?", (to,)).fetchone()
        if active_user_row:
            active_user = dict(active_user_row)
            messages = db.execute("""
                SELECT * FROM messages
                WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
                ORDER BY created_at ASC
            """, (user["id"], to, to, user["id"])).fetchall()
            messages = [dict(m) for m in messages]

            # Mark incoming messages as read
            db.execute("""
                UPDATE messages SET is_read = 1
                WHERE sender_id = ? AND receiver_id = ? AND is_read = 0
            """, (to, user["id"]))
            # Mark related message notifications as read
            db.execute("""
                UPDATE notifications SET is_read = 1
                WHERE user_id = ? AND type = 'message' AND related_user_id = ?
            """, (user["id"], to))
            db.commit()

    db.close()

    return templates.TemplateResponse("messages.html", {
        "request": request,
        "user": dict(user),
        "conversations": conversations,
        "active_user": active_user,
        "messages": messages,
    })


@app.post("/messages/send")
async def send_message(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False, "error": "Not authenticated"}, status_code=401)

    try:
        body = await request.json()
        receiver_id = int(body.get("receiver_id"))
        content = body.get("content", "").strip()
        if not content:
            return JSONResponse({"success": False, "error": "Empty message"}, status_code=400)

        db = get_db()
        receiver = db.execute("SELECT id FROM users WHERE id = ?", (receiver_id,)).fetchone()
        if not receiver:
            db.close()
            return JSONResponse({"success": False, "error": "User not found"}, status_code=404)

        db.execute(
            "INSERT INTO messages (sender_id, receiver_id, content) VALUES (?, ?, ?)",
            (user["id"], receiver_id, content)
        )
        db.commit()
        msg = db.execute("SELECT * FROM messages WHERE id = last_insert_rowid()").fetchone()

        # Create a notification for the receiver with message preview
        preview = content if len(content) <= 50 else content[:47] + "…"
        notif_content = f"{user['name']} sent you a message: \"{preview}\""
        db.execute("""
            INSERT INTO notifications (user_id, type, content, related_user_id)
            VALUES (?, 'message', ?, ?)
        """, (receiver_id, notif_content, user["id"]))
        db.commit()

        db.close()
        return JSONResponse({"success": True, "message": dict(msg)})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/messages/poll")
async def poll_messages(request: Request, to: int, after: int = 0):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"messages": []})

    db = get_db()
    # Fetch all new messages in this conversation (both directions)
    new_msgs = db.execute("""
        SELECT * FROM messages
        WHERE ((sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?))
          AND id > ?
        ORDER BY created_at ASC
    """, (to, user["id"], user["id"], to, after)).fetchall()

    # Mark incoming messages as read
    db.execute("""
        UPDATE messages SET is_read = 1
        WHERE sender_id = ? AND receiver_id = ? AND is_read = 0
    """, (to, user["id"]))
    db.execute("""
        UPDATE notifications SET is_read = 1
        WHERE user_id = ? AND type = 'message' AND related_user_id = ?
    """, (user["id"], to))
    db.commit()
    db.close()
    return JSONResponse({"messages": [dict(m) for m in new_msgs]})


@app.get("/messages/conversations/poll")
async def poll_conversations(request: Request):
    """Lightweight endpoint to refresh sidebar conversation list."""
    user = get_current_user(request)
    if not user:
        return JSONResponse({"conversations": []})

    db = get_db()
    partner_rows = db.execute("""
        SELECT DISTINCT other_user_id FROM (
            SELECT receiver_id AS other_user_id FROM messages WHERE sender_id = ?
            UNION
            SELECT sender_id AS other_user_id FROM messages WHERE receiver_id = ?
        )
    """, (user["id"], user["id"])).fetchall()

    conversations = []
    for row in partner_rows:
        other_id = row["other_user_id"]
        other_user = db.execute("SELECT id, name, avatar_path FROM users WHERE id = ?", (other_id,)).fetchone()
        if not other_user:
            continue
        last_msg = db.execute("""
            SELECT content, created_at FROM messages
            WHERE (sender_id = ? AND receiver_id = ?) OR (sender_id = ? AND receiver_id = ?)
            ORDER BY created_at DESC LIMIT 1
        """, (user["id"], other_id, other_id, user["id"])).fetchone()
        unread = db.execute("""
            SELECT COUNT(*) as cnt FROM messages
            WHERE sender_id = ? AND receiver_id = ? AND is_read = 0
        """, (other_id, user["id"])).fetchone()["cnt"]

        c = last_msg["content"] if last_msg else ""
        conversations.append({
            "other_user_id": other_id,
            "other_name": other_user["name"],
            "other_avatar": other_user["avatar_path"] or "",
            "last_message": (c[:40] + "…") if len(c) > 40 else c,
            "last_time": last_msg["created_at"] if last_msg else "",
            "unread_count": unread,
        })

    conversations.sort(key=lambda c: (0 if c["unread_count"] > 0 else 1, c["last_time"] or ""), reverse=False)
    db.close()
    return JSONResponse({"conversations": conversations})


# ====================== FOLLOWERS / FOLLOWING API ======================

@app.get("/api/followers/{user_id}")
async def get_followers(user_id: int, request: Request):
    current_user = get_current_user(request)
    current_id = current_user["id"] if current_user else 0

    db = get_db()
    rows = db.execute("""
        SELECT u.id, u.name, u.avatar_path,
            (SELECT status FROM follows
             WHERE follower_id = ? AND following_id = u.id) as my_follow_status
        FROM follows f
        JOIN users u ON f.follower_id = u.id
        WHERE f.following_id = ? AND f.status = 'accepted'
        ORDER BY f.created_at DESC
    """, (current_id, user_id)).fetchall()
    db.close()
    return JSONResponse([dict(r) for r in rows])


@app.get("/api/following/{user_id}")
async def get_following(user_id: int, request: Request):
    current_user = get_current_user(request)
    current_id = current_user["id"] if current_user else 0

    db = get_db()
    rows = db.execute("""
        SELECT u.id, u.name, u.avatar_path,
            (SELECT status FROM follows
             WHERE follower_id = ? AND following_id = u.id) as my_follow_status
        FROM follows f
        JOIN users u ON f.following_id = u.id
        WHERE f.follower_id = ? AND f.status = 'accepted'
        ORDER BY f.created_at DESC
    """, (current_id, user_id)).fetchall()
    db.close()
    return JSONResponse([dict(r) for r in rows])


# ====================== NOTIFICATIONS API ======================

@app.get("/api/notifications")
async def get_notifications_api(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"notifications": [], "unread": 0})

    db = get_db()
    notifs = db.execute("""
        SELECT n.*, u.name as sender_name, u.avatar_path as sender_avatar
        FROM notifications n
        LEFT JOIN users u ON n.related_user_id = u.id
        WHERE n.user_id = ?
        ORDER BY n.created_at DESC
        LIMIT 30
    """, (user["id"],)).fetchall()

    unread = db.execute(
        "SELECT COUNT(*) as c FROM notifications WHERE user_id = ? AND is_read = 0",
        (user["id"],)
    ).fetchone()["c"]

    db.close()
    return JSONResponse({
        "notifications": [dict(n) for n in notifs],
        "unread": unread
    })


@app.post("/api/notifications/read")
async def mark_notifications_read(request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False})

    db = get_db()
    db.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (user["id"],))
    db.commit()
    db.close()
    return JSONResponse({"success": True})


# ====================== STORIES API ======================

@app.post("/stories/upload")
async def upload_story(
    request: Request,
    image: UploadFile = File(...),
    caption: str = Form("")
):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False, "error": "Not authenticated"}, status_code=401)

    try:
        ext = image.filename.split(".")[-1].lower()
        if ext not in ["jpg", "jpeg", "png", "webp"]:
            return JSONResponse({"success": False, "error": "Invalid file type"}, status_code=400)

        fname = f"story_{uuid.uuid4().hex}.{ext}"
        image_path = f"uploads/{fname}"
        with open(image_path, "wb") as f:
            shutil.copyfileobj(image.file, f)

        expires_at = (datetime.now(timezone.utc) + timedelta(hours=24)).strftime("%Y-%m-%d %H:%M:%S")

        db = get_db()
        db.execute(
            "INSERT INTO stories (user_id, image_path, caption, expires_at) VALUES (?, ?, ?, ?)",
            (user["id"], image_path, caption.strip(), expires_at)
        )
        db.commit()
        story = db.execute("SELECT * FROM stories WHERE id = last_insert_rowid()").fetchone()
        db.close()

        return JSONResponse({"success": True, "story": dict(story)})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/api/stories")
async def get_stories_api(request: Request):
    user = get_current_user(request)
    current_id = user["id"] if user else 0

    db = get_db()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    stories = db.execute("""
        SELECT s.*, u.name as user_name, u.avatar_path,
            (SELECT COUNT(*) FROM story_views WHERE story_id = s.id AND viewer_id = ?) as is_viewed
        FROM stories s
        JOIN users u ON s.user_id = u.id
        WHERE s.expires_at > ?
        ORDER BY s.user_id = ? DESC, s.created_at DESC
    """, (current_id, now, current_id)).fetchall()

    db.close()
    return JSONResponse([dict(s) for s in stories])


@app.post("/api/stories/{story_id}/view")
async def mark_story_viewed(story_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False})

    db = get_db()
    try:
        db.execute(
            "INSERT OR IGNORE INTO story_views (story_id, viewer_id) VALUES (?, ?)",
            (story_id, user["id"])
        )
        db.commit()
    except Exception:
        pass
    db.close()
    return JSONResponse({"success": True})


@app.delete("/api/stories/{story_id}")
async def delete_story(story_id: int, request: Request):
    user = get_current_user(request)
    if not user:
        return JSONResponse({"success": False}, status_code=401)

    db = get_db()
    story = db.execute("SELECT * FROM stories WHERE id = ? AND user_id = ?", (story_id, user["id"])).fetchone()
    if not story:
        db.close()
        return JSONResponse({"success": False, "error": "Not found or not yours"}, status_code=404)

    db.execute("DELETE FROM story_views WHERE story_id = ?", (story_id,))
    db.execute("DELETE FROM stories WHERE id = ?", (story_id,))
    db.commit()
    db.close()
    return JSONResponse({"success": True})