// gcc -Wall -Wextra -pedantic -shared -fPIE -fPIC -o IM.so immediate_gfx.c -lSDL2 -lSDL2_ttf -lSDL2_image
#include "immediate_gfx.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <time.h>
#include <unistd.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <SDL2/SDL_image.h>

#define MAX_KEY_EVENTS_PER_FRAME 32
#define MAX_INPUT_LENGTH 4096

const struct { int sym; const char *name; } KEYCODE_TO_NAME[] = {
    { SDLK_q, "q" }, { SDLK_w, "w" }, { SDLK_e, "e" }, { SDLK_r, "r" }, { SDLK_t, "t" }, { SDLK_y, "y" }, { SDLK_u, "u" }, { SDLK_i, "i" }, { SDLK_o, "o" },
    { SDLK_p, "p" }, { SDLK_a, "a" }, { SDLK_s, "s" }, { SDLK_d, "d" }, { SDLK_f, "f" }, { SDLK_g, "g" }, { SDLK_h, "h" }, { SDLK_j, "j" }, { SDLK_k, "k" },
    { SDLK_l, "l" }, { SDLK_z, "z" }, { SDLK_x, "x" }, { SDLK_c, "c" }, { SDLK_v, "v" }, { SDLK_b, "b" }, { SDLK_n, "n" }, { SDLK_m, "m" }, { SDLK_1, "1" },
    { SDLK_2, "2" }, { SDLK_3, "3" }, { SDLK_4, "4" }, { SDLK_5, "5" }, { SDLK_6, "6" }, { SDLK_7, "7" }, { SDLK_8, "8" }, { SDLK_9, "9" }, { SDLK_0, "0" },
    { SDLK_SPACE, "space" }, { SDLK_LEFT, "left arrow" }, { SDLK_RIGHT, "right arrow" }, { SDLK_UP, "up arrow" }, { SDLK_DOWN, "down arrow" },
    { SDLK_BACKSPACE, "backspace" }, { SDLK_RETURN, "enter" }, { SDLK_DELETE, "delete" },
    { SDLK_UNKNOWN, NULL }
};

static TTF_Font *bubble_font, *input_font;
static int window_width, window_height, window_fps, window_running, initialized;
static struct timespec last_frame;
static struct im_key_event key_events[MAX_KEY_EVENTS_PER_FRAME];
static unsigned int n_key_events;
static SDL_Window *window;
SDL_Renderer *renderer;

void im_init(void)
{
    // Create window, create renderer, load fonts
    if (!(initialized++)) {
        SDL_SetMainReady();

        TTF_Init();
        SDL_Init(SDL_INIT_VIDEO);
        if (SDL_CreateWindowAndRenderer(1, 1, SDL_WINDOW_HIDDEN, &window, &renderer))
            abort();
        bubble_font = TTF_OpenFont("/usr/share/fonts/TTF/liberation/LiberationSans-Regular.ttf", 14);  // TODO: Don't hardcode this
        if (!bubble_font)
            abort();
    }
    input_font = TTF_OpenFont("calibril.ttf", 12);
    if (!input_font)
        abort();
}

void im_quit(void)
{
    if (!(--initialized)) {
        TTF_CloseFont(bubble_font);
        TTF_CloseFont(input_font);
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        TTF_Quit();
        IMG_Quit();
        SDL_Quit();
    }
}

void im_set_context(int width, int height, int fps)
{
    SDL_SetWindowSize(window, width, height);
    SDL_ShowWindow(window);
    window_width = width;
    window_height = height;
    window_fps = fps;
    clock_gettime(CLOCK_MONOTONIC, &last_frame);
}

void im_mouse_pos(int *x, int *y)
{
    SDL_GetMouseState(x, y);
}

void *im_load_texture(const char *data, size_t len, int *restrict w, int *restrict h)
{
    SDL_RWops *src = SDL_RWFromMem((void*)data, len);
    SDL_Surface *surf = IMG_Load_RW(src, 1);
    if (!surf)
        abort();
    if (w)
        *w = surf->w;
    if (h)
        *h = surf->h;
    SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
    if (!tex)
        abort();
    SDL_FreeSurface(surf);
    return tex;
}

uint8_t *im_load_mask(const char *data, size_t len, int *restrict w, int *restrict h)
{
    SDL_RWops *src = SDL_RWFromMem((void*)data, len);
    SDL_Surface *surf = IMG_Load_RW(src, 1);
    if (!surf)
        abort();
    if (w)
        *w = surf->w;
    if (h)
        *h = surf->h;
    SDL_Surface *surf2 = SDL_ConvertSurfaceFormat(surf, SDL_PIXELFORMAT_RGBA32, 0);
    if (!surf2)
        abort();
    SDL_FreeSurface(surf);
    surf = surf2;
    SDL_LockSurface(surf);
    uint8_t *mask = calloc((surf->w * surf->h / 8 / sizeof(uint8_t) + 1), sizeof(uint8_t));
    for (int pixelidx = 0; pixelidx < surf->w * surf->h; ++pixelidx)
        mask[pixelidx >> 3] |= (((uint8_t*)surf->pixels)[pixelidx*4+3] > 127) << (pixelidx % 8);
    SDL_UnlockSurface(surf);
    SDL_FreeSurface(surf);
    return mask;
}

void im_rotated_rectangle_extents(double w, double h, double angle, int *restrict outw, int *restrict outh)
{
    angle = (angle - 90.0) * (double)M_PI / 180.0;

    int corners[4][2] = { { -w/2, -h/2 }, { -w/2, +h/2 }, { +w/2, +h/2 }, { +w/2, -h/2 } };
    int minx = w, maxx = 0, miny = h, maxy = 0;
    for (int i = 0; i < 4; ++i) {
        int cornerx = corners[i][0], cornery = corners[i][1];
        int newx = cornerx * cos(angle) - cornery * sin(angle), newy = cornerx * sin(angle) + cornery * cos(angle);
        if (newx < minx) minx = newx;
        if (newx > maxx) maxx = newx;
        if (newy < miny) miny = newy;
        if (newy > maxy) maxy = newy;
    }
    *outw = maxx - minx;
    *outh = maxy - miny;
}

void im_draw_texture(void *texture, int x, int y, double size, double angle, double anchorx, double anchory)
{
    int w, h;
    SDL_QueryTexture(texture, NULL, NULL, &w, &h);
    w *= size;
    h *= size;
    SDL_Rect rect = { (int)(-anchorx * w + x + window_width / 2), (int)(-anchory * h - y + window_height / 2), w, h };
    SDL_Point center = { (int)(anchorx * w), (int)(anchory * h) };
    SDL_RenderCopyEx(renderer, (SDL_Texture*)texture, NULL, &rect, angle - 90.0, &center, SDL_FLIP_NONE);
}

void im_free_texture(void *texture)
{
    if (initialized)
        SDL_DestroyTexture((SDL_Texture*)texture);
}

uint8_t *im_tex_to_mask(void *texture, double angle, double scale, int *restrict wout, int *restrict hout)
{
    int w, h, dstw, dsth;
    SDL_QueryTexture(texture, NULL, NULL, &w, &h);
    im_rotated_rectangle_extents((double)w * scale, (double)h * scale, angle, &dstw, &dsth);
    if (wout)
        *wout = dstw;
    if (hout)
        *hout = dsth;
    if (dstw == 0 || dsth == 0)
        return NULL;

    SDL_Texture *result = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET, dstw, dsth);
    if (!result)
        abort();
    SDL_SetTextureBlendMode(result, SDL_BLENDMODE_NONE);
    SDL_SetRenderTarget(renderer, result);
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 0);
	SDL_RenderClear(renderer);
    SDL_Rect rect = { (int)(dstw-w*scale)/2, (int)(dsth-h*scale)/2, (int)(w*scale), (int)(h*scale) };
	SDL_RenderCopyEx(renderer, (SDL_Texture*)texture, NULL, &rect, angle - 90.0, NULL, SDL_FLIP_NONE);
    rect = (SDL_Rect){ 0, 0, dstw, dsth };
    uint8_t *pxs = malloc(dstw * dsth * sizeof(uint32_t));
    if (!pxs)
        abort();

    if (SDL_RenderReadPixels(renderer, &rect, SDL_PIXELFORMAT_RGBA32, pxs, dstw * sizeof(uint32_t)))
        abort();  // slow?

    SDL_SetRenderTarget(renderer, NULL);
    SDL_DestroyTexture(result);

    uint8_t *mask = calloc(dstw * dsth / 8 + 1, 1);
    if (!mask)
        abort();
    for (int pixelidx = 0; pixelidx < dstw * dsth; ++pixelidx)
        mask[pixelidx >> 3] |= (pxs[pixelidx*4+3] > 127) << (pixelidx % 8);
    free(pxs);

    return mask;
}

void im_begin_frame(void)
{
    window_running = 1;

    // Keep constant FPS
    struct timespec cur_frame;
    clock_gettime(CLOCK_MONOTONIC, &cur_frame);
    long nsec = cur_frame.tv_nsec - last_frame.tv_nsec + (cur_frame.tv_sec - last_frame.tv_sec) * 1000000000L;
    if (nsec >= 0) {
        nsec = 1000000000L/window_fps - nsec;
        if (nsec > 0)
            usleep(nsec / 1000);
    }
    clock_gettime(CLOCK_MONOTONIC, &last_frame);

    // Fill background
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_Rect rect = { 0, 0, window_width, window_height };
    SDL_RenderFillRect(renderer, &rect);

    // Collect events
    n_key_events = 0;
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (event.type == SDL_KEYDOWN || event.type == SDL_KEYUP) {
            int i;
            for (i = 0; KEYCODE_TO_NAME[i].sym != SDLK_UNKNOWN; ++i)
                if (KEYCODE_TO_NAME[i].sym == event.key.keysym.sym)
                    break;
            key_events[n_key_events].keyname = KEYCODE_TO_NAME[i].name;
            key_events[n_key_events++].held = event.type == SDL_KEYDOWN;
        } else if (event.type == SDL_QUIT) {
            window_running = 0;
        }
    }
}

struct im_key_event *im_get_key_events(size_t *nevents)
{
    if (nevents)
        *nevents = n_key_events;
    return key_events;
}

void im_end_frame(void)
{
    SDL_RenderPresent(renderer);
}

void im_stop(void)
{
    window_running = 0;
}

int im_running(void)
{
    return window_running;
}

void im_draw_line(unsigned r, unsigned g, unsigned b, int x1, int y1, int x2, int y2, int sw)
{
    SDL_SetRenderDrawColor(renderer, r, g, b, 255);
    double length = sqrt((x2-x1)*(x2-x1) + (y2-y1)*(y2-y1));
    double xd = (y2-y1) / length, yd = -(x2-x1) / length;
    for (int i = -sw / 2; i < sw - (-sw / 2); ++i)
        SDL_RenderDrawLine(renderer, (int)(x1 + xd * i), (int)(y1 + yd * i), (int)(x2 + xd * i), (int)(y2 + yd * i));
}

void im_draw_rect_fill(unsigned r, unsigned g, unsigned b, int x, int y, int w, int h, int bw)
{
    if (bw * 2 > w || bw * 2 > h)
        return;

    SDL_SetRenderDrawColor(renderer, r, g, b, 255);
    if (bw > 0) {
        SDL_Rect rect = { x+bw, y, w-bw*2, bw };
        SDL_RenderFillRect(renderer, &rect);
        rect = (SDL_Rect){ x, y+bw, w, h-bw*2 };
        SDL_RenderFillRect(renderer, &rect);
        rect = (SDL_Rect){ x+bw, y+h-bw, w-bw*2, bw };
        SDL_RenderFillRect(renderer, &rect);

        im_draw_circle_fill(r, g, b, x + bw,     y + bw,     bw, 0, 0, 1, 0);
        im_draw_circle_fill(r, g, b, x + w - bw, y + bw,     bw, 0, 0, 0, 1);
        im_draw_circle_fill(r, g, b, x + bw,     y + h - bw, bw, 1, 0, 0, 0);
        im_draw_circle_fill(r, g, b, x + w - bw, y + h - bw, bw, 0, 1, 0, 0);
    } else {
        SDL_Rect rect = { x, y, w, h };
        SDL_RenderFillRect(renderer, &rect);
    }
}

void im_draw_rect_stroke(unsigned r, unsigned g, unsigned b, int x, int y, int w, int h, int sw, int bw)
{
    if (bw * 2 > w || bw * 2 > h)
        return;

    SDL_SetRenderDrawColor(renderer, r, g, b, 255);
    for (int i = 0; i < sw; ++i) {
        SDL_RenderDrawLine(renderer, x+bw, y, x+w-bw, y);
        SDL_RenderDrawLine(renderer, x+bw, y+h-1, x+w-bw, y+h-1);
        SDL_RenderDrawLine(renderer, x, y+bw, x, y+h-bw);
        SDL_RenderDrawLine(renderer, x+w-1, y+bw, x+w-1, y+h-bw);
        if (bw > 0) {
            im_draw_circle(r, g, b, x + bw,     y + h - bw, bw, 1, 0, 0, 0);
            im_draw_circle(r, g, b, x + w - bw, y + h - bw, bw, 0, 1, 0, 0);
            im_draw_circle(r, g, b, x + bw,     y + bw,     bw, 0, 0, 1, 0);
            im_draw_circle(r, g, b, x + w - bw, y + bw,     bw, 0, 0, 0, 1);
        }
        x++;
        y++;
        w -= 2;
        h -= 2;
    }
}

void im_draw_circle(unsigned r, unsigned g, unsigned b, int x, int y, int rad, int bl, int br, int tl, int tr)
{
    SDL_SetRenderDrawColor(renderer, r, g, b, 255);
    int px = rad - 1, py = 0, tx = 1, ty = 1, err = tx - rad * 2;
    do {
        if (tl) {
            SDL_RenderDrawPoint(renderer, x - px, y - py);
            SDL_RenderDrawPoint(renderer, x - py, y - px);
        }
        if (bl) {
            SDL_RenderDrawPoint(renderer, x - px, y + py);
            SDL_RenderDrawPoint(renderer, x - py, y + px);
        }
        if (tr) {
            SDL_RenderDrawPoint(renderer, x + px, y - py);
            SDL_RenderDrawPoint(renderer, x + py, y - px);
        }
        if (br) {
            SDL_RenderDrawPoint(renderer, x + px, y + py);
            SDL_RenderDrawPoint(renderer, x + py, y + px);
        }

        if (err <= 0) {
            ty += 2;
            err += ty;
            py++;
        } else {
            tx += 2;
            err += tx - rad * 2;
            px--;
        }
    } while (py <= px);
}

void im_draw_circle_stroke(unsigned r, unsigned g, unsigned b, int x, int y, int rad, int sw, int bl, int br, int tl, int tr)
{
    for (int i = 0; i < sw; ++i)
        im_draw_circle(r, g, b, x, y, rad-i, bl, br, tl, tr);
}

void im_draw_circle_fill(unsigned r, unsigned g, unsigned b, int x, int y, int rad, int bl, int br, int tl, int tr)
{
    for (int i = 0; i < rad; ++i)
        im_draw_circle(r, g, b, x, y, rad-i, bl, br, tl, tr);
}

void im_draw_thought_bubble(const char *input_text, int x, int y)
{
    int input_length = strlen(input_text), max_width = 0;
    char *text = calloc(input_length + 1, 1);
    unsigned prev_ws = 0, lineno = 0;
    SDL_Surface *lines[64];
    for (int i = 0, j = 0; i < input_length + 1;) {
        char cur = input_text[i++];
        if ((cur & 248) == 240) {
            text[j++] = cur;
            text[j++] = input_text[i++];
            text[j++] = input_text[i++];
            text[j++] = input_text[i++];
        } else if ((cur & 240) == 224) {
            text[j++] = cur;
            text[j++] = input_text[i++];
            text[j++] = input_text[i++];
        } else if ((cur & 224) == 192) {
            text[j++] = cur;
            text[j++] = input_text[i++];
        } else if (cur == ' ' || cur == '\t' || cur == '\n') {
            if (prev_ws) {
                continue;
            } else {
                text[j++] = ' ';
                prev_ws = 1;
            }
        } else {
            prev_ws = 0;
            text[j++] = cur;
        }
        int w, h;
        if (TTF_SizeUTF8(bubble_font, text, &w, &h))
            abort();
        if (w > 108 || i == input_length) {
            int whitespace = j-1;
            for (; whitespace > 0; --whitespace)
                if (text[whitespace] == ' ')
                    break;
            if (whitespace)
                text[whitespace] = 0;

            if (TTF_SizeUTF8(bubble_font, text, &w, &h))
                abort();
            if (w > max_width)
                max_width = w;

            SDL_Surface *surf = TTF_RenderUTF8_Blended(bubble_font, text, (SDL_Color){ 0, 0, 0, 255 });
            if (!surf)
                abort();
            if (whitespace) {
                memmove(text, text + whitespace + 1, input_length - whitespace - 1);
                memset(text + whitespace, 0, input_length + 1 - whitespace);
                j = j - whitespace - 1;
            } else {
                memset(text, 0, input_length + 1);
                j = 0;
            }
            // TODO: Allow an arbitrary number of lines to be drawn
            if (lineno < sizeof(lines) / sizeof(lines[0]))
                lines[lineno++] = surf;
        } else if (w > max_width)
            max_width = w;
    }
    free(text);

    int bubble_width = 32 + max_width, bubble_height = 41 + lineno * 22;
    if (bubble_width < 55)
        bubble_width = 55;

    x = x + 240 - bubble_width * 3 / 4;
    y = -y + 180 - bubble_height + 7;

    im_draw_rect_fill(255, 255, 255, x + 4, y + 4, bubble_width - 4, bubble_height - 24, 12);
    im_draw_rect_stroke(220, 220, 220, x + 4, y + 4, bubble_width - 4, bubble_height - 24, 2, 12);

    im_draw_circle_fill(255, 255, 255, x + bubble_width/2+2, y + bubble_height-20, 8, 1, 1, 1, 1);
    im_draw_circle_stroke(220, 220, 220, x + bubble_width/2+2, y + bubble_height-22, 8, 2, 1, 1, 0, 0);
    im_draw_circle_fill(255, 255, 255, x + bubble_width/2+9, y + bubble_height-13, 4, 1, 1, 1, 1);
    im_draw_circle_stroke(220, 220, 220, x + bubble_width/2+9, y + bubble_height-13, 4, 2, 1, 1, 1, 1);
    im_draw_circle_fill(255, 255, 255, x + bubble_width/2+16, y + bubble_height-10, 3, 1, 1, 1, 1);
    im_draw_circle_stroke(220, 220, 220, x + bubble_width/2+16, y + bubble_height-10, 3, 2, 1, 1, 1, 1);

    for (unsigned i = 0; i < lineno; ++i) {
        SDL_Surface *surf = lines[i];
        SDL_Rect dst = { x + 16, y + 16 + i * 22, surf->w, surf->h };
        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
        if (!tex)
            abort();
        SDL_FreeSurface(surf);
        SDL_RenderCopy(renderer, tex, NULL, &dst);
        SDL_DestroyTexture(tex);
    }
}

int im_fast_overlap(const uint8_t *restrict mask1, int w1, int h1, const uint8_t *restrict mask2, int w2, int h2, int xoffs, int yoffs, int *restrict xout, int *restrict yout)
{
    // TODO: Faster implementation pls; ideally, we'd be checking 64 (or maybe even 256) bits at a time
    // return 0;
    int maxx1 = 0 > xoffs ? 0 : xoffs, maxy1 = 0 > yoffs ? 0 : yoffs;
    int minx2 = w1 < xoffs+w2 ? w1 : xoffs+w2, miny2 = h1 < yoffs+h2 ? h1 : yoffs+h2;

    // SDL_SetRenderDrawColor(renderer, 0, 255, 0, 255);
    // SDL_Rect rect = { maxx1, maxy1, minx2-maxx1, miny2-maxy1 };
    // SDL_RenderDrawRect(renderer, &rect);

    for (int y = maxy1; y < miny2; ++y) {
        for (int x = maxx1; x < minx2; ++x) {
            int idx1 = y*w1 + x, idx2 = (y-yoffs)*w2 + (x-xoffs);
            if ((mask1[idx1 >> 3] >> (idx1 % 8)) & (mask2[idx2 >> 3] >> (idx2 % 8)) & 1) {
                if (xout)
                    *xout = x;
                if (yout)
                    *yout = y;
                return 1;
            }
        }
    }
    return 0;
}

uint8_t *im_empty_mask(int width, int height, int dft)
{
    uint8_t *mask = calloc(width * height / 8 + 1, 1);
    if (!mask)
        abort();
    memset(mask, dft ? 255 : 0, width * height / 8 + 1);
    return mask;
}

// TODO: Use TEXTINPUT it does everything for you (right now you don't have shift etc)
struct im_text_input_state *im_draw_question_box(struct im_text_input_state *state)
{
    if (!state) {
        state = calloc(sizeof *state, 1);
        state->text = calloc(MAX_INPUT_LENGTH + 1, 1);
        state->closed = 0;
        state->cursor_location = 0;
    }

    int mousex;
    int mousey; // that's me!
    int pressed = SDL_GetMouseState(&mousex, &mousey) & SDL_BUTTON(1);
    
    if (pressed && mousex >= 27 && mousex <= 328 && mousey >= 428 && mousey <= 460) {
        // User clicked on the editable area
        state->cursor_location = 0;
        clock_gettime(CLOCK_MONOTONIC, &state->edit_start);
    }

    if (state->cursor_location != -1) {
        for (unsigned i = 0; i < n_key_events; ++i) {
            clock_gettime(CLOCK_MONOTONIC, &state->edit_start);
            if ((!key_events[i].keyname) || (!key_events[i].held))
                continue;
            if (key_events[i].keyname[1]) {
                if (strcmp(key_events[i].keyname, "backspace") == 0) {
                    if (state->cursor_location > 0) {
                        int len = state->cursor_location < 1 ? 0 : state->cursor_location - 1;
                        memmove(state->text + len, state->text + state->cursor_location, strlen(state->text) - state->cursor_location + 1);
                        --state->cursor_location;
                    }

                } else if (strcmp(key_events[i].keyname, "delete") == 0) {
                    int len = state->cursor_location;
                    if (state->text[len])
                        memmove(state->text + len, state->text + state->cursor_location + 1, strlen(state->text) - state->cursor_location);
                
                } else if (strcmp(key_events[i].keyname, "left arrow") == 0) {
                    if (state->cursor_location > 0)
                        --state->cursor_location;
                
                } else if (strcmp(key_events[i].keyname, "right arrow") == 0) {
                    if (state->cursor_location < (int)strlen(state->text))
                        ++state->cursor_location;
                
                } else if (strcmp(key_events[i].keyname, "enter") == 0) {
                    state->closed = 1;

                } else if (strcmp(key_events[i].keyname, "space") == 0) {
                    if (strlen(state->text) < MAX_INPUT_LENGTH) {
                        memmove(state->text + state->cursor_location + 1, state->text + state->cursor_location, strlen(state->text) - state->cursor_location + 1);
                        state->text[state->cursor_location++] = ' ';
                    }
                }
            } else {
                if (strlen(state->text) < MAX_INPUT_LENGTH) {
                    memmove(state->text + state->cursor_location + 1, state->text + state->cursor_location, strlen(state->text) - state->cursor_location + 1);
                    state->text[state->cursor_location++] = key_events[i].keyname[0];
                }
            }
        }
    }

    im_draw_rect_fill(255, 255, 255, 7, 284, 468, 66, 10);
    im_draw_rect_stroke(210, 210, 210, 7, 284, 468, 66, 2, 10);

    im_draw_rect_fill(255, 255, 255, 27, 301, 428, 32, 15);
    im_draw_rect_stroke(210, 210, 210, 27, 301, 428, 32, 1, 15);

    im_draw_circle_fill(135, 80, 195, 438, 317, 13, 1, 1, 1, 1);

    im_draw_line(255, 255, 255, 432, 322, 429, 318, 5);
    im_draw_line(255, 255, 255, 433, 322, 441, 314, 5);
    im_draw_circle_fill(255, 255, 255, 429, 318, 2, 1, 1, 1, 1);
    im_draw_circle_fill(255, 255, 255, 433, 323, 2, 1, 1, 1, 1);

    im_draw_circle_fill(255, 255, 255, 442, 314, 2, 1, 1, 1, 1);

    if (state->text[0] != 0) {
        SDL_Surface *surf = TTF_RenderUTF8_Blended(input_font, state->text, (SDL_Color){ 50, 50, 50, 255 });
        if (!surf)
            abort();
        SDL_Texture *tex = SDL_CreateTextureFromSurface(renderer, surf);
        if (!tex)
            abort();
        SDL_Rect dst = { 38, 311, surf->w, surf->h };
        SDL_FreeSurface(surf);
        SDL_RenderCopy(renderer, tex, NULL, &dst);
        SDL_DestroyTexture(tex);
    }

    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (state->cursor_location != -1 && (now.tv_sec - state->edit_start.tv_sec) % 2 == 0) {
        // font.size(state.text[:state.cursor_location])[0]
        char c = state->text[state->cursor_location];
        state->text[state->cursor_location] = 0;
        int w, h;
        if (TTF_SizeUTF8(input_font, state->text, &w, &h))
            abort();
        state->text[state->cursor_location] = c;
        im_draw_line(0, 0, 0, 38 + w, 311, 38 + w, 321, 1);
    }

    return state;
}
