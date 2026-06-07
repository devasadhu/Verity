#include "hashmap.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

/* FNV-1a hash — fast, good distribution for string keys */
static uint64_t fnv1a(const char *key) {
    uint64_t hash = 0xcbf29ce484222325ULL;
    while (*key) {
        hash ^= (unsigned char)*key++;
        hash *= 0x100000000001b3ULL;
    }
    return hash;
}

static int is_empty(const HMSlot *slot) {
    return slot->key == NULL;
}

static int is_tombstone(const HMSlot *slot) {
    return slot->key == HASHMAP_TOMBSTONE;
}

static int is_live(const HMSlot *slot) {
    return slot->key != NULL && slot->key != HASHMAP_TOMBSTONE;
}

HashMap *hm_create(size_t initial_capacity) {
    if (initial_capacity == 0) initial_capacity = HASHMAP_INITIAL_CAPACITY;

    HashMap *hm = malloc(sizeof(HashMap));
    if (!hm) return NULL;

    hm->slots = calloc(initial_capacity, sizeof(HMSlot));
    if (!hm->slots) { free(hm); return NULL; }

    hm->capacity   = initial_capacity;
    hm->count      = 0;
    hm->tombstones = 0;
    return hm;
}

void hm_destroy(HashMap *hm) {
    if (!hm) return;
    for (size_t i = 0; i < hm->capacity; i++) {
        if (is_live(&hm->slots[i])) {
            free(hm->slots[i].key);
            free(hm->slots[i].value);
        }
    }
    free(hm->slots);
    free(hm);
}

/* Resize and rehash into a new table of given capacity */
static int hm_resize(HashMap *hm, size_t new_capacity) {
    HMSlot *new_slots = calloc(new_capacity, sizeof(HMSlot));
    if (!new_slots) return -1;

    for (size_t i = 0; i < hm->capacity; i++) {
        if (!is_live(&hm->slots[i])) continue;

        uint64_t h = fnv1a(hm->slots[i].key);
        size_t idx = h % new_capacity;

        /* Linear probe into new table (no tombstones in new table) */
        while (new_slots[idx].key != NULL) {
            idx = (idx + 1) % new_capacity;
        }
        new_slots[idx].key   = hm->slots[i].key;   /* transfer ownership */
        new_slots[idx].value = hm->slots[i].value;
    }

    free(hm->slots);
    hm->slots      = new_slots;
    hm->capacity   = new_capacity;
    hm->tombstones = 0;
    return 0;
}

int hm_set(HashMap *hm, const char *key, const char *value) {
    /* Resize if load factor (live + tombstones) exceeds threshold */
    double load = (double)(hm->count + hm->tombstones) / (double)hm->capacity;
    if (load >= HASHMAP_MAX_LOAD_FACTOR) {
        if (hm_resize(hm, hm->capacity * 2) != 0) return -1;
    }

    uint64_t h   = fnv1a(key);
    size_t   idx = h % hm->capacity;
    size_t   tombstone_idx = SIZE_MAX;  /* first tombstone we pass */

    while (1) {
        HMSlot *slot = &hm->slots[idx];

        if (is_empty(slot)) {
            /* Insert here, or at earliest tombstone */
            size_t insert_idx = (tombstone_idx != SIZE_MAX) ? tombstone_idx : idx;
            HMSlot *ins = &hm->slots[insert_idx];

            ins->key   = strdup(key);
            ins->value = strdup(value);
            if (!ins->key || !ins->value) return -1;

            hm->count++;
            if (tombstone_idx != SIZE_MAX) hm->tombstones--;
            return 0;
        }

        if (is_tombstone(slot)) {
            if (tombstone_idx == SIZE_MAX) tombstone_idx = idx;
        } else if (strcmp(slot->key, key) == 0) {
            /* Update existing key */
            free(slot->value);
            slot->value = strdup(value);
            return slot->value ? 0 : -1;
        }

        idx = (idx + 1) % hm->capacity;
    }
}

const char *hm_get(const HashMap *hm, const char *key) {
    uint64_t h   = fnv1a(key);
    size_t   idx = h % hm->capacity;

    while (1) {
        const HMSlot *slot = &hm->slots[idx];

        if (is_empty(slot)) return NULL;  /* key definitely not present */

        if (is_live(slot) && strcmp(slot->key, key) == 0) {
            return slot->value;
        }

        idx = (idx + 1) % hm->capacity;
    }
}

int hm_delete(HashMap *hm, const char *key) {
    uint64_t h   = fnv1a(key);
    size_t   idx = h % hm->capacity;

    while (1) {
        HMSlot *slot = &hm->slots[idx];

        if (is_empty(slot)) return 0;  /* not found */

        if (is_live(slot) && strcmp(slot->key, key) == 0) {
            free(slot->key);
            free(slot->value);
            slot->key   = HASHMAP_TOMBSTONE;
            slot->value = NULL;
            hm->count--;
            hm->tombstones++;
            return 1;
        }

        idx = (idx + 1) % hm->capacity;
    }
}

size_t hm_count(const HashMap *hm)    { return hm->count; }
size_t hm_capacity(const HashMap *hm) { return hm->capacity; }