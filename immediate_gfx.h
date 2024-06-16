#ifndef IMMEDIATE_GFX_H
#define IMMEDIATE_GFX_H

#include <stdlib.h>
#include <stdint.h>

#define MAX_KEY_EVENTS_PER_FRAME 32

struct im_key_event {
    const char *keyname;
    int held;
};

struct im_text_input_state
{
    char *text;
    int cursor_location;
    struct timespec edit_start;
    int closed;
};

#include <SDL2/SDL.h>
extern SDL_Renderer *renderer;

void im_init(void);
void im_quit(void);
void im_set_context(int width, int height, int fps);

void im_mouse_pos(int *x, int *y);

void im_rotated_rectangle_extents(double w, double h, double angle, int *restrict outw, int *restrict outh);

void *im_load_texture(const char *data, size_t len, int *restrict w, int *restrict h);
void im_draw_texture(void *texture, int x, int y, double size, double angle, double anchorx, double anchory);
void im_free_texture(void *texture);

void im_begin_frame(void);
struct im_key_event *im_get_key_events(size_t *nevents);
void im_end_frame(void);

void im_stop(void);
int im_running(void);

void im_draw_line         (unsigned r, unsigned g, unsigned b, int x1, int y1, int x2, int y2, int sw);

void im_draw_rect_fill    (unsigned r, unsigned g, unsigned b, int x, int y, int w, int h,         int bw);
void im_draw_rect_stroke  (unsigned r, unsigned g, unsigned b, int x, int y, int w, int h, int sw, int bw);

void im_draw_circle       (unsigned r, unsigned g, unsigned b, int x, int y, int rad,         int bl, int br, int tl, int tr);
void im_draw_circle_fill  (unsigned r, unsigned g, unsigned b, int x, int y, int rad,         int bl, int br, int tl, int tr);
void im_draw_circle_stroke(unsigned r, unsigned g, unsigned b, int x, int y, int rad, int sw, int bl, int br, int tl, int tr);

uint8_t *im_empty_mask(int width, int height, int dft);
uint8_t *im_load_mask(const char *data, size_t len, int *restrict w, int *restrict h);
uint8_t *im_tex_to_mask(void *texture, double angle, double scale, int *restrict wout, int *restrict hout);
int im_fast_overlap(const uint8_t *restrict mask1, int w1, int h1, const uint8_t *restrict mask2, int w2, int h2, int xoffs, int yoffs, int *restrict xout, int *restrict yout);

struct im_text_input_state *im_draw_question_box(struct im_text_input_state *state);
void im_draw_thought_bubble(const char *input_text, int x, int y);

#endif  // IMMEDIATE_GFX_H
