#ifndef HASHMAP_H
#define HASHMAP_H

#include <stdint.h>
#include <stddef.h>

#define HASHMAP_INITIAL_CAPACITY 64
#define HASHMAP_MAX_LOAD_FACTOR  0.70
#define HASHMAP_TOMBSTONE        ((char *)-1)  /* sentinel for deleted slots */

typedef struct {
    char    *key;       /* heap-allocated copy, NULL = empty, TOMBSTONE = deleted */
    char    *value;     /* heap-allocated copy */
} HMSlot;

typedef struct {
    HMSlot   *slots;
    size_t    capacity;
    size_t    count;     /* live entries */
    size_t    tombstones;
} HashMap;

HashMap *hm_create(size_t initial_capacity);
void     hm_destroy(HashMap *hm);

/* Returns 0 on success, -1 on OOM */
int      hm_set(HashMap *hm, const char *key, const char *value);

/* Returns value string (caller must not free), or NULL if not found */
const char *hm_get(const HashMap *hm, const char *key);

/* Returns 1 if key existed and was deleted, 0 otherwise */
int      hm_delete(HashMap *hm, const char *key);

size_t   hm_count(const HashMap *hm);
size_t   hm_capacity(const HashMap *hm);

#endif /* HASHMAP_H */